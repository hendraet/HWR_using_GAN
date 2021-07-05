from sys import argv
from .main_torch_latest import BATCH_SIZE,  sort_batch, HIDDEN_SIZE_DEC, HIDDEN_SIZE_ENC, EMBEDDING_SIZE, TRADEOFF_CONTEXT_EMBED, Bi_GRU, CON_STEP, crit, learning_rate, lr_milestone, lr_gamma
import subprocess as sub
from .loadData2_vgg import IAM_words
from .utils import writePredict, HEIGHT, WIDTH, output_max_len, vocab_size, FLIP
import time
import torch
from torch import optim
from .models.encoder_vgg import Encoder
from .models.decoder import Decoder
from .models.attention import locationAttention as Attention
from .models.seq2seq import Seq2Seq
import os
import config

EPOCH = 5
log_softmax = torch.nn.LogSoftmax(dim=-1)


def writeLoss(loss_value, author, flag, n_synth_imgs):
    folder_name = 'results/'
    file_name = folder_name + f'/{author}/{n_synth_imgs}_imgs_{flag}_loss'
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'a') as f:
        f.write(str(loss_value))
        f.write('\n')


def train(train_loader,seq2seq, opt, epoch, author):
    
    seq2seq.train()
    total_loss = 0
    for num, (train_index, train_in, train_in_len, train_out) in enumerate(train_loader):
        #train_in = train_in.unsqueeze(1)
        train_in, train_out = train_in.cuda(), train_out.cuda()
        output, attn_weights = seq2seq(train_in, train_out, train_in_len, teacher_rate=0, train=True) # (100-1, 32, 62+1)
        writePredict(epoch, train_index, output, f'author_{author}_train')
        train_label = train_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
        output_l = output.view(-1, vocab_size) # remove last <EOS>
      
        loss = crit(log_softmax(output_l.view(-1, vocab_size)), train_label)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.data.item()

    total_loss /= (num+1)
    return total_loss


def train_with_synthesized_imgs(author, n_synth_imgs, config):
    # base_dir = 'parser/'
    filename = f'HWR_train_partitions/partition_{author}'
    image_folder = f'synthesized_images/{author}/'

    with open(filename, 'r') as f_tr:
        data = f_tr.readlines()
        file_labels = [i[:-1].split(' ') for i in data]

    data_set = IAM_words(file_labels, image_folder, augmentation=False)
    data_loader = torch.utils.data.DataLoader(data_set, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=0, pin_memory=True)

    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    model_file = config['hwr_default_model']
    print('Loading ' + model_file)
    seq2seq.load_state_dict(torch.load(model_file)) #load
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)

    # for i in range(EPOCH +1):
    #     scheduler.step()

    test_loader = get_test_loader(author)
    start_epoch = 0

    test_loss = test(test_loader, seq2seq, start_epoch, author)
    print(f'epoch: {start_epoch} test_loss: {test_loss}')
    writeLoss(test_loss, author, 'test', n_synth_imgs)

    for epoch in range(EPOCH):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]  # TODO: needed?
        train_loss = train(data_loader, seq2seq, opt, epoch + 1, author)
        print(f'epoch: {epoch + 1} train_loss: {train_loss}')
        writeLoss(train_loss, author, 'train', n_synth_imgs)
        test_loss = test(test_loader, seq2seq, epoch + 1, author)
        print(f'epoch: {epoch + 1} test_loss: {test_loss}')
        writeLoss(test_loss, author, 'test', n_synth_imgs)


def get_test_loader(author):
    # base_dir = 'parser/'
    filename = f'HWR_Groundtruth/gt_{author}'

    with open(filename, 'r') as f_tr:
        data = f_tr.readlines()
        file_labels = [i[:-1].split(' ') for i in data]

    data_set = IAM_words(file_labels, augmentation=False)
    data_loader = torch.utils.data.DataLoader(data_set, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=0, pin_memory=True)
    return data_loader


def test(test_loader, seq2seq, model, author):
    seq2seq.eval()
    total_loss_t = 0
    for num, (test_index, test_in, test_in_len, test_out) in enumerate(test_loader):
        with torch.no_grad():
            test_in, test_out = test_in.cuda(), test_out.cuda()
            output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)
        writePredict(model, test_index, output_t, f'author_{author}')
        test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)

        output_l = output_t.view(-1, vocab_size)  # remove last <EOS>

        loss = crit(log_softmax(output_l.view(-1, vocab_size)), test_label)
        total_loss_t += loss.data.item()

    total_loss_t /= (num + 1)
    return total_loss_t


def predict(model, author):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    model_file = 'save_weights/seq2seq-' + str(model) +'.model'
    print('Loading ' + model_file)
    seq2seq.load_state_dict(torch.load(model_file)) #load
    test(get_test_loader(author), seq2seq, model, author)


def calculate_cer(model, author):
    base_dir = 'pred_logs/'
    labels = f'parser/OCR_test_{author}.filter27'
    predictions = f'{base_dir}author_{author}_predict_seq.{str(model)}.log'
    cer = sub.Popen(['./tasas_cer.sh', labels, predictions], stdout=sub.PIPE)
    cer = cer.stdout.read().decode('utf8')
    return float(cer)/100



if __name__ == '__main__':
    author = int(argv[1])
    # author = 560
    print(time.ctime())
    # predict(model, author)
    # cer = calculate_cer(model, author)
    # print(cer)

    train_with_synthesized_imgs(author)