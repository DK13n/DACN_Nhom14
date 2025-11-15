# Dồ án chuyên ngành - Nhóm 14 (Face Liveness / Spoof Checker)

Dự án cung cấp API backend bằng Python và một giao diện frontend trực quan giúp demo khả năng phân biệt khuôn mặt thật và khuôn mặt giả (spoof) từ ảnh hoặc video.Dùng để xác thực người dùng khi đăng nhập hoặc truy cập vào các dịch vụ nội bộ trong công ty, phòng lab hoặc trung tâm nghiên cứu.

## ✨ Tính năng chính
//...Đang cập nhật

## 🛠 Công nghệ sử dụng
//...Đang cập nhật

---
## 🧱 Cấu trúc thư mục
```
project_root/
├── fe/
│ └── index.html # Giao diện Frontend (HTML + JS) dùng để gửi yêu cầu đến API
│
├── pvcore/ # Mã nguồn backend chính
│ ├── main.py # Điểm khởi chạy API (FastAPI)
│ ├── config.py # Cấu hình hệ thống: đường dẫn dữ liệu, tham số model, seed,...
│ ├── api/
│ │ ├── init.py
│ │ └── routers/
│ │ └── init.py # Khai báo & gom nhóm các route API
│ ├── models/
│ │ ├── init.py
│ │ └── weights/
│ │ └── init.py # Thư mục lưu trọng số mô hình
│ └── shared/
│ └── init.py # Module chứa các hàm tiện ích dùng chung
│
├── notebooks/
│ └── README.md # Notebook / ghi chú phát triển
├── sever/ # Thư mục dự phòng (hiện trống)
│
└── pyproject.toml # Metadata dự án và khai báo dependencies

```
---
## 📦 Cài đặt & chạy dự án
//...Đang cập nhật
