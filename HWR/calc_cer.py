import subprocess as sub

base_dir = '../pred_logs/'
label_file_tim = '../test_samples_tim/labels.txt'
label_file_washington = '../washington_test/labels.txt'

def calculate_cer(labels_file, run_id):
    predictions_test = f'{base_dir}id_{run_id}_test_predict_seq.log'
    predictions_original = f'{base_dir}id_{run_id}_original_test_predict_seq.log'
    cer_test = sub.Popen(['./tasas_cer.sh', labels_file, predictions_test], stdout=sub.PIPE)
    cer_test = float(cer_test.stdout.read().decode('utf8'))/100
    cer_original = sub.Popen(['./tasas_cer.sh', labels_file, predictions_original], stdout=sub.PIPE)
    cer_original = float(cer_original.stdout.read().decode('utf8')) / 100
    with open(f'{base_dir}{run_id}_test.cer', 'w') as f:
        f.write(str(cer_test))
    with open(f'{base_dir}{run_id}_original.cer', 'w') as f:
        f.write(str(cer_original))



if __name__ == '__main__':
    for i in range(123, 223):
        calculate_cer(label_file_tim, i)

    for i in range(223, 323):
        calculate_cer(label_file_washington, i)

    for i in range(327, 427):
        calculate_cer(label_file_washington, i)

