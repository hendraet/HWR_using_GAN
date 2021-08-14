from .main_torch_latest import BATCH_SIZE, sort_batch, HIDDEN_SIZE_DEC, HIDDEN_SIZE_ENC, EMBEDDING_SIZE, \
    TRADEOFF_CONTEXT_EMBED, Bi_GRU, CON_STEP, crit, learning_rate, lr_milestone, lr_gamma
from .loadData2_vgg import IAM_words
from .utils import writePredict, HEIGHT, WIDTH, output_max_len, vocab_size, FLIP
import torch
from torch import optim
from .models.encoder_vgg import Encoder
from .models.decoder import Decoder
from .models.attention import locationAttention as Attention
from .models.seq2seq import Seq2Seq
import os
from pathlib import Path
import config

# TODO Move this to config/args
EPOCH = 1


def init_s2s_model(model_file=config.hwr_default_model):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    print(f'Loading {model_file}')
    seq2seq.load_state_dict(torch.load(model_file))  # load
    return seq2seq


# TODO From utils
def write_loss(loss_value, wid, flag, n_synth_imgs, folder_name='results'):
    # TODO Move hard-coded path
    file_name = Path(folder_name, wid, f'{n_synth_imgs}_imgs_{flag}_loss')
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'a') as f:
        f.write(f'{str(loss_value)} \n')


def calc_loss(output, labels):
    test_label = labels.permute(1, 0)[1:].contiguous().view(-1)  # remove<GO>
    output_label = output.view(-1, vocab_size)  # remove last <EOS>
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = crit(log_softmax(output_label), test_label)
    return loss


def train(train_loader, seq2seq, opt, prediction_path=None, file_name=None):
    seq2seq.train()
    total_loss = 0
    for num, (train_index, train_in, train_in_len, train_out) in enumerate(train_loader):
        train_in, train_out = train_in.cuda(), train_out.cuda()
        output, attention_weights = seq2seq(train_in, train_out, train_in_len,
                                            teacher_rate=0, train=True)  # (100-1, 32, 62+1)

        if prediction_path and file_name:
            writePredict(prediction_path, file_name, train_index, output)

        loss = calc_loss(output, train_out)
        total_loss += loss.data.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    total_loss /= (num + 1)
    return total_loss


def test(test_loader, seq2seq, prediction_path, file_name):
    seq2seq.eval()
    total_loss = 0
    for num, (test_index, test_in, test_in_len, test_out) in enumerate(test_loader):
        with torch.no_grad():
            test_in, test_out = test_in.cuda(), test_out.cuda()
            output, attention_weights = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)

        writePredict(prediction_path, file_name, test_index, output)
        loss = calc_loss(output, test_out)
        total_loss += loss.data.item()

    total_loss /= (num + 1)
    return total_loss


def train_with_synth_imgs_from_input_folder(model_id, labels_file, image_folder):
    data_loader = get_test_loader(labels_file, image_folder)
    seq2seq = init_s2s_model()
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)

    for epoch in range(EPOCH):
        scheduler.step()
        train_loss = train(data_loader, seq2seq, opt)
        print(f'epoch: {epoch + 1} train_loss: {train_loss}')

    folder_weights = f'final_weights_HWR/{model_id}/'
    if not os.path.exists(folder_weights):
        os.makedirs(folder_weights)
    torch.save(seq2seq.state_dict(), f'{folder_weights}/seq2seq-{model_id}.model')
    print(f'Trained model saved as {model_id}.')


def test_with_imgs_from_input_folder(labels_file, image_folder, model_path, prediction_file_prefix):
    # TODO Move hard-coded path
    prediction_path = 'pred_logs'
    # prediction_file_prefix = f'id_{id}_'

    test_loader = get_test_loader(labels_file, image_folder)
    seq2seq = init_s2s_model(model_path)
    test_loss = test(test_loader, seq2seq, prediction_path, f'{prediction_file_prefix}test_predict_seq')
    print(f'Test loss of {model_path}: {test_loss}')


def train_and_test_with_synthesized_imgs(author, n_synth_imgs, labels_file, image_folder):
    # TODO Move hard-coded path
    prediction_path = 'pred_logs'
    prediction_file_prefix = f'author_{author}_'

    data_loader = get_test_loader(labels_file, image_folder)
    seq2seq = init_s2s_model()
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)

    # for i in range(EPOCH +1):
    #     scheduler.step()

    test_loader = get_test_loader(f'HWR_Groundtruth/gt_{author}')
    start_epoch = 0

    test_loss = test(test_loader, seq2seq, prediction_path, f'{prediction_file_prefix}test_predict_seq.{start_epoch}')
    print(f'epoch: {start_epoch} test_loss: {test_loss}')
    write_loss(test_loss, author, 'test', n_synth_imgs)

    for epoch in range(EPOCH):
        scheduler.step()
        train_loss = train(data_loader, seq2seq, opt, prediction_path,
                           f'{prediction_file_prefix}train_predict_seq.{epoch + 1}')
        print(f'epoch: {epoch + 1} train_loss: {train_loss}')
        write_loss(train_loss, author, 'train', n_synth_imgs)
        test_loss = test(test_loader, seq2seq, prediction_path, f'{prediction_file_prefix}test_predict_seq.{epoch + 1}')
        print(f'epoch: {epoch + 1} test_loss: {test_loss}')
        write_loss(test_loss, author, 'test', n_synth_imgs)


# TODO Move hard-coded path
def get_test_loader(filename, image_folder=None):
    # base_dir = 'parser/'

    with open(filename, 'r') as f_tr:
        data = f_tr.readlines()
        file_labels = [i[:-1].split(' ') for i in data]

    if image_folder:
        data_set = IAM_words(file_labels, image_folder, augmentation=False)
    else:
        data_set = IAM_words(file_labels, augmentation=False)
    data_loader = torch.utils.data.DataLoader(data_set, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=0, pin_memory=True)
    return data_loader


# TODO Remove if unused, else move hard-coded path
def predict(model, author):
    model_file = 'save_weights/seq2seq-' + str(model) + '.model'
    seq2seq = init_s2s_model(model_file)
    # print('Loading ' + model_file)
    test(get_test_loader(author), seq2seq, model, author)
