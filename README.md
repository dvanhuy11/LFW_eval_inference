# LFW_compare  
## LFW Download
```bash
http://vis-www.cs.umass.edu/lfw/#download
```
## Link enrollment.json 
```bash
https://drive.google.com/file/d/1gIHuy0OE-dd06MujYM9TDlnYo0-si--O/view?usp=drive_link
```
## Pipeline   
- Tạo tập đăng ký từ những hình ảnh đầu tiên của từng người ( bao gồm nhận diện khuôn mặt [mtcnn] , và rút trích đặc trưng sau đó embedding )
- So sánh:
  - Duyệt qua từng folder_person: Những hình ảnh chưa được đăng ký sẽ được chọn để so sánh.   
  - Nhận diện khuôn mặt, rút trích đặc trưng và embedding.
  - Tính Cosine của ảnh đó với tập đăng ký ( 1-n ) . Có nghĩa là lấy hình ảnh hiện tại để thực hiện phép tính cosine với từng người đã được thêm vào tập đăng ký.
## Scripts
### Create Enrollment
Tất cả được chạy trên server 7gb GPU và 31gb CPU
```bash
python3 /media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/util/create_enrollment.py
```
### Compare 
```bash
python3 /media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/util/eval_LFW.py
```
