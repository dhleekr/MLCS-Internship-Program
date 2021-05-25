import os
import csv


path_to_train_dataset_1 = os.path.join('datasets','mnist_train.csv')
path_to_train_dataset_2 = os.path.join('datasets','mnist_train_2.csv')
# path_to_test_dataset = os.path.join('datasets','mnist_test.csv')

# paths_to_datasets = os.path.join('datasets','*.csv') # Returns list of paths

output_path_train = os.path.join('outputs', 'train')
output_path_test = os.path.join('outputs', 'test')

if not os.path.exists(output_path_train):
    os.makedirs(output_path_train)
if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)


row_index = 0
with open(path_to_train_dataset_1, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        print(row)
        label = row[0]
        output_file = open(os.path.join('outputs/train', '{}-{}.csv'.format(label,row_index)),'w')
        csv_writer = csv.writer(output_file)
        for i in range(28):
            start = i*28 + 1
            finish = start + 28
            csv_writer.writerow(row[start:finish])
        row_index += 1

with open(path_to_train_dataset_2, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        print(row)
        label = row[0]
        output_file = open(os.path.join('outputs/train', '{}-{}.csv'.format(label,row_index)),'w')
        csv_writer = csv.writer(output_file)
        for i in range(28):
            start = i*28 + 1
            finish = start + 28
            csv_writer.writerow(row[start:finish])
        row_index += 1
            
output_file.close