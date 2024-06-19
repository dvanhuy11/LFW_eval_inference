# LFW_compare  
- Quá trình đánh giá diễn ra 6:40 phút
![image](https://github.com/dvanhuy11/LFW_compare/assets/76443374/09fa0c6f-e745-4e99-b62c-237da6c4583f)

## LFW Download
```bash
http://vis-www.cs.umass.edu/lfw/#download
```
## Link enrollment.json 
```bash
https://drive.google.com/file/d/1gIHuy0OE-dd06MujYM9TDlnYo0-si--O/view?usp=drive_link
```
## Link kết quả đánh giá LFW
```bash
https://drive.google.com/file/d/1DY9YbUVzC3G4_RS-9rhDsuJOyJTQY3lQ/view?usp=drive_link
```
## Link kết quả đánh giá Dlib
```bash
https://drive.google.com/file/d/1v8pmbwhjOWNN2-bh4kxFHe9xLztUBDDw/view?usp=drive_link
```
## Pipeline   
- Tạo tập đăng ký từ những hình ảnh đầu tiên của từng người ( bao gồm nhận diện khuôn mặt [MTCNN] , và rút trích đặc trưng [VGGFace2] sau đó embedding )
- Lưu ý khi dùng MTCNN và VGGFace2
```bash
   pip install tensorflow==2.12.1                                                            
   pip install keras==2.12                                                                   
                                                                                             
   I solved this issue by changing the import from                                           
      from keras.engine.topology import get_source_inputs                                    
   to                                                                                        
      from keras.utils.layer_utils import get_source_inputs in keras_vggface/models.py.
```
 
- So sánh:
  - Duyệt qua từng folder_person: Những hình ảnh chưa được đăng ký sẽ được chọn để so sánh.   
  - Nhận diện khuôn mặt, rút trích đặc trưng và embedding.
  - Tính Cosine của ảnh đó với tập đăng ký ( 1-n ) . Có nghĩa là lấy hình ảnh hiện tại để thực hiện phép tính cosine với từng người đã được thêm vào tập đăng ký.
## Scripts
### Create Enrollment
Tất cả được chạy trên server VAST.HPC
```bash
python3 /media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/util/create_enrollment.py
```
### Compare 
```bash
python3 /media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/util/eval_LFW.py
```
