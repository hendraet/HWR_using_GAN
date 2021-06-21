import subprocess as sub
# import "../predict"


train_partition = open('parser/train_OCR_560.part', 'r+')

data = train_partition.read()

# print(data)

data = data.split()

train_partition.close()


# result = exec(open('predict.py').read())
res_cer = sub.call(['python3', 'predict.py', '45'], stdout=sub.PIPE)
result = res_cer.stdout.read().decode('utf8')


print(result)






# train_partition.write(data[100:])
#
# train_partition.close()










