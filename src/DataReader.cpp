#include "DataReader.h"
using namespace std;

DataReader::DataReader (string train_data_add, string test_data_add) {

}

vector<int> DataReader::src_data_reading(MatrixXd& tr_i, MatrixXd& tr_l, string image, string label,MatrixXi& test_true,string type) {
    ifstream file_image(image,ios::binary);
    int number_of_images = 0;
    int number_of_rows = 0;
    int number_of_columns = 0;
    if (file_image.is_open()) {
      int magic_number = 0;
      file_image.read((char*)&magic_number,sizeof(magic_number));
      magic_number = bigToEndian(magic_number);
      file_image.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images = bigToEndian(number_of_images);
      file_image.read((char*)&number_of_rows,sizeof(number_of_rows));
      number_of_rows = bigToEndian(number_of_rows);
      file_image.read((char*)&number_of_columns,sizeof(number_of_columns));
      number_of_columns = bigToEndian(number_of_columns);
      for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < number_of_rows; j++) {
          for (int k = 0; k < number_of_columns; k++) {
            unsigned char value = 0;
            file_image.read((char*)&value,sizeof(value));
            tr_i(i,number_of_rows*j+k) = value/255.0f;
          }
        }
      }
      file_image.close();
    }
    ifstream file_label(label,ios::binary);
    if (file_label.is_open()) {
      int magic_number = 0;
      int number_of_images = 0;
      file_label.read((char*)&magic_number,sizeof(magic_number));
      magic_number = bigToEndian(magic_number);
      file_label.read((char*)&number_of_images,sizeof(number_of_images));
      number_of_images = bigToEndian(number_of_images);
      for (int i = 0; i < number_of_images; i++) {
        unsigned char value = 0;
        file_label.read((char*)&value,sizeof(value));
        tr_l(i,value) = 1;
        if (type == "test") test_true(i) = value;
      }
      file_label.close();
    }
    return {number_of_images, number_of_rows*number_of_columns};
}
int DataReader::bigToEndian (int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}