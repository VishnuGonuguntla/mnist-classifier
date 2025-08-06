#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <string>
using namespace std;
using Eigen::MatrixXf, Eigen::MatrixXd;
int BigEndiantoEndian (int i) {
    unsigned char c1, c2, c3, c4;
    c1=i&255;
    c2=(i>>8)&255;
    c3=(i>>16)&255;
    c4=(i>>24)&255;
    return((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}
void ReadMNIST(int NumberOfImages, MatrixXf& label_data,string source) {
    ifstream file (source,ios::binary);
    if (file.is_open()) {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= BigEndiantoEndian(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= BigEndiantoEndian(number_of_images);
        for(int i=0;i<number_of_images;++i) {
            uint8_t temp=0;
            file.read((char*)&temp,sizeof(temp));
            label_data(i,temp)= 1.0f;
        }
    } else { cout<<"Unable to open file" << endl; }
    file.close();
}
int main(int argc, const char * argv[]) {
    string source = argv[1];
    string output = argv[2];
    int index = atoi(argv[3]);
    MatrixXf label_data(60000,10);
    ReadMNIST(60000,label_data,source);
    ofstream file2(output);
        if (file2.is_open()) {
        file2 << 1 << endl;
        file2 << 10 << endl;
        for (int i = 2; i < 12; i++) {
          file2<< label_data(index,i-2) <<endl;
        }
        file2.close();
    } else { cout<<"Unable to open file" << endl; }
    return 0;
}