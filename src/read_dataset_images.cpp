#include <iostream>
#include <fstream>
#include <Eigen/Dense>
using Eigen::MatrixXf, Eigen::MatrixXd;
using namespace std;
int BigEndiantoEndian (int i) {
    unsigned char c1, c2, c3, c4;
    c1=i&255;
    c2=(i>>8)&255;
    c3=(i>>16)&255;
    c4=(i>>24)&255;
    return((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}
void ReadMNIST(MatrixXf& pixel_data,string source) {
    ifstream file (source,ios::binary);
    if (file.is_open()) {
        int magic_number=0, number_of_images=0, number_of_rows=0, number_of_columns=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= BigEndiantoEndian(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= BigEndiantoEndian(number_of_images);
        file.read((char*)&number_of_rows,sizeof(number_of_rows));
        number_of_rows= BigEndiantoEndian(number_of_rows);
        file.read((char*)&number_of_columns,sizeof(number_of_columns));
        number_of_columns= BigEndiantoEndian(number_of_columns);
        // Pixel Data Allocation
        for(int i=0;i<number_of_images;++i) {
            for(int r=0;r<number_of_rows;++r) {
                for(int c=0;c<number_of_columns;++c) {
                    uint8_t temp=0;
                    file.read((char*)&temp, sizeof(temp));
                    pixel_data(i,(number_of_rows*r)+c)= temp/255.0f;
                }
            }
        }
        file.close();
    }
    else { cout<<"Error opening file1"<<endl; }
}
int main(int argc, const char * argv[]) {
    string source = argv[1], output = argv[2];
    int index = atoi(argv[3]);
    MatrixXf pixel_data = MatrixXf::Zero(60000,784);
    MatrixXd out_file = MatrixXd::Zero(787,1);
    ReadMNIST(pixel_data,source);
    ofstream file(output);
    if (file.is_open()) {
        file << 2 << endl;
        file << 28 << endl;
        file << 28 << endl;
          for (int i = 0; i<784; i++) {
              file << pixel_data(index,i) << endl;
          }
        file.close();
    } else { cout<<"Error opening file";}
    return 0;
}