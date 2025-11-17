# Dồ án chuyên ngành - Nhóm 14 (Face Liveness / Spoof Checker)

Dự án cung cấp API backend bằng Python và một giao diện frontend trực quan giúp demo khả năng phân biệt khuôn mặt thật và khuôn mặt giả (spoof) từ ảnh hoặc video.Dùng để xác thực người dùng khi đăng nhập hoặc truy cập vào các dịch vụ nội bộ trong công ty, phòng lab hoặc trung tâm nghiên cứu.

![Frontend Demo](pvcore/shared/Images/image.png)

## ✨ Tính năng chính
- **Huấn luyện**: Model Hybrid-CDCN-ReSVIT tự thiết kế (tự build backbone + head). và model Transfer learning với MobileNetV3 trích xuất đặc trưng.
- **Frontend**: `fe/index.html` dùng JS “bridge” kết nối backend (CORS đã bật, chỉ cần gọi đúng các endpoint trên là chạy được).
  
## 🛠 Công nghệ sử dụng
- **Python 3.9+** – Ngôn ngữ chính xây dựng backend & training  
- **FastAPI** – Xây dựng REST API phục vụ huấn luyện và suy luận  
- **Uvicorn** – Web server chạy ứng dụng FastAPI  
- **PyTorch** – Framework deep learning cho Hybrid-CDCN-ReSVIT & MobileNetV3  
- **TorchVision** – Pretrained MobileNetV3 + transform ảnh  
- **OpenCV / Pillow** – Đọc ảnh & tiền xử lý ảnh đầu vào  
- **HTML / CSS / JavaScript** – Frontend đơn giản kết nối API  
- **CORS Middleware** – Cho phép frontend giao tiếp backend  

### **Hybrid-CDCN-ReSVIT (Tự build)**
- **CDCN – Central Difference Convolution Network**  
  Sử dụng các phép vi phân trung tâm để nhạy với texture và gradient.
- **Vision Transformer (ViT) hoặc ReSViT – Residual Swin-based ViT**  
  Khai thác cơ chế self-attention để học quan hệ toàn cục.
- **Custom Fusion (CNN + ViT)**  
  Kết hợp đặc trưng cục bộ (CNN) và toàn cục (ViT) thành một kiến trúc hybrid.

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

 **Đầu tiên clone dự án về local** : `git clone https://github.com/DK13n/DACN_Nhom14.git`

 Cách 1. **Tạo venv & cài deps**
   ```bash
   # Windows PowerShell ,Linux/WSL:
   # Nếu bạn chua ở trong thư mục DACN_Nhom14
   cd DACN_Nhom14
   source Scripts/run_be.sh  
   ```
  -Sau đó mở `fe/index.html` bằng **Live Server** (VS Code) → FE gọi `http://127.0.0.1:8000`.

 cách 2 : 
 Tạo môi trường .vev 
 ```bash
 uv sync
#Linux / Ubutu
 source .venv/bin/activate 
 ```
 ```bash
 uv sync
#Window
 .venv/scripts/activate 
 ```
 Chạy be và fe.
  **Chạy backend**
   ```bash
   uvicorn pvcore.main:app --host 0.0.0.0 --port 8000 --reload
   ```
  **Mở frontend**
   - Mở `fe/index.html` bằng **Live Server** (VS Code) → FE gọi `http://127.0.0.1:8000`.




