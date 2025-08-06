#include <fstream>
#include <Eigen/Dense>
#include "../include/DataReader.h"

using namespace std;
using Eigen::MatrixXd, Eigen::MatrixXi;

DataReader::DataReader (map<string,string> config_data) {
  train_i = config_data["rel_path_train_images"];
  train_l = config_data["rel_path_train_labels"];
  test_i = config_data["rel_path_test_images"];
  test_l = config_data["rel_path_test_labels"];
}

vector<int> DataReader::deCompress(MatrixXd& image, MatrixXi& label, string type) {
    ifstream file_image;
    if (type == "train") { file_image.open(train_i,ios::binary); } else {file_image.open(test_i,ios::binary);}
    int magic_number = 0;
    int number_of_images = 0;
    int number_of_rows = 0;
    int number_of_columns = 0;
    // Image Data Reading
    if (file_image.is_open()) {
      file_image.read((char*)&magic_number,sizeof(magic_number));
      magic_number = bigToEndian(magic_number);
      file_image.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images = bigToEndian(number_of_images);
      file_image.read((char*)&number_of_rows,sizeof(number_of_rows));
      number_of_rows = bigToEndian(number_of_rows);
      file_image.read((char*)&number_of_columns,sizeof(number_of_columns));
      number_of_columns = bigToEndian(number_of_columns);
      number_of_rows = 28;
      number_of_columns = 28;
      // MatrixXd image(number_of_images, 28*28);
      for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < 28; j++) {
          for (int k = 0; k < 28; k++) {
            unsigned char value = 0;
            file_image.read((char*)&value,sizeof(value));
            image(i, k + number_of_rows*j) = value/255.0f;
          }
        }
      }
      file_image.close();
      // ofstream image_out;
      // if (type == "train") { image_out.open("../mnist-datasets/train_image.txt",ios::binary); } else {image_out.open("../mnist-datasets/test_image.txt",ios::binary);}
      // image_out << image << endl;
      // image_out.close();

    }
    // Label Data Reading:
    ifstream file_label;
    cout << train_l << endl; cout << test_l << endl;
    if (type == "train") { file_label.open(train_l,ios::binary); } else {file_label.open(test_l,ios::binary);}
    if (file_label.is_open()) {
      file_label.read((char*)&magic_number,sizeof(magic_number));
      magic_number = bigToEndian(magic_number);
      file_label.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images = bigToEndian(number_of_images);
      // MatrixXi label = MatrixXi::Zero(number_of_images, 10);
      for (int i = 0; i < number_of_images; i++) {
        unsigned char value = 0;
        file_label.read((char*)&value,sizeof(value));
        label(i, value) = 1;
      }
      file_label.close();
      // ofstream label_out;
      // if (type == "train") { label_out.open("../mnist-datasets/train_label.txt",ios::binary); } else { label_out.open("../mnist-datasets/test_label.txt",ios::binary);}
      // label_out<<label;
      // label_out.close();
    } else {cout <<"Not open" << endl;}

    return {number_of_images, number_of_rows*number_of_columns, 10};
}
int DataReader::bigToEndian (int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}