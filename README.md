# FRA — Face Recognition Attendance

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat-square&logo=mysql&logoColor=white)
![insightface](https://img.shields.io/badge/insightface-0.7.3-FF6B6B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Hệ thống điểm danh nhận diện khuôn mặt theo thời gian thực**
sử dụng SCRFD + ArcFace/ResNet-50, xây dựng trên Flask + MySQL.

[Tính năng](#-tính-năng) · [Cài đặt](#-cài-đặt) · [Sử dụng](#-sử-dụng) · [API](#-api) · [Kiến trúc](#-kiến-trúc)

</div>

---

## 📌 Giới thiệu

**FRA (Face Recognition Attendance)** là hệ thống điểm danh tự động ứng dụng deep learning:

- **SCRFD** phát hiện khuôn mặt và trả về bounding box + 5 landmarks
- **ArcFace / ResNet-50** trích xuất embedding 512 chiều, L2-chuẩn hóa
- **Cosine Similarity** so sánh embedding với gallery để xác định danh tính
- **Ngưỡng**: ≥ 0.70 → Điểm danh · 0.50–0.70 → Không chắc · < 0.50 → Thất bại

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 🎥 **Nhận diện realtime** | Phát hiện đồng thời ≤ 3 khuôn mặt/frame, vẽ bounding box + tên lên camera |
| 📷 **Đăng ký nhiều ảnh** | Chụp 1–10 ảnh/người, tính embedding trung bình để tăng độ chính xác |
| ➕ **Thêm ảnh sau** | Thêm ảnh cho sinh viên đã đăng ký, hệ thống tự tính lại vector đại diện |
| 📋 **Điểm danh tự động** | Ghi nhận lần đầu mỗi người mỗi ngày (confidence ≥ 70%) |
| 📊 **Admin Dashboard** | Biểu đồ điểm danh thành công + phân loại tất cả lần quét theo ngày |
| 🗂️ **Nhật ký đầy đủ** | Lưu **mọi** lần quét (cả thất bại) vào `recognition_logs` |
| 🔍 **Bộ lọc nâng cao** | Lọc nhật ký theo kết quả, khoảng thời gian, tên sinh viên |
| 🔐 **Phân quyền Admin** | Attendance, Register, Admin yêu cầu đăng nhập |
| 📦 **Batch enroll LFW** | Script đăng ký hàng loạt từ dataset LFW-deepfunneled |

---

## 🏗️ Kiến trúc

```
Browser (WebRTC Camera)
        │  base64 JPEG frame
        ▼
Flask Server (Waitress WSGI)
  ├── SCRFD       → detect faces, bounding box + 5 landmarks
  ├── Align       → affine warp → 112×112 crop
  ├── ResNet-50   → feature map [B, 2048, 7, 7] → GAP → [B, 2048]
  ├── ArcFace     → FC + BN + L2-norm → embedding [B, 512]
  └── Cosine Sim  → match vs gallery → name + confidence
        │
        ▼
MySQL Server
  ├── students            (embedding trung bình mỗi người)
  ├── student_embeddings  (từng ảnh riêng lẻ — audit trail)
  ├── attendance          (1 bản ghi / người / ngày)
  └── recognition_logs    (mọi lần quét, cả thất bại)
```

### Ngưỡng nhận diện

```
sim ≥ 0.70  →  ✅ Confident  →  Ghi nhận điểm danh
sim  0.50–0.70  →  ⚠️  Uncertain  →  Hiển thị cảnh báo góc/ánh sáng
sim < 0.50  →  ❌ Unknown   →  Không nhận diện
```

---

## 📁 Cấu trúc thư mục

```
FRA-main/
├── app.py                        # Entry point (Waitress WSGI)
├── config.py                     # Cấu hình DB, model, ngưỡng
├── requirements.txt
│
├── database/
│   └── db.py                     # Kết nối MySQL, khởi tạo bảng
│
├── models/
│   ├── student_model.py          # CRUD sinh viên + multi-embedding
│   └── attendance_model.py       # Điểm danh + recognition_logs
│
├── services/
│   ├── face_service.py           # Pipeline SCRFD → ArcFace
│   ├── attendance_service.py     # Orchestrator: detect → log → mark
│   └── embedding_cache.py        # In-memory gallery cache
│
├── routes/
│   ├── camera_routes.py          # /recognize, /enroll-multi, /enroll-add-photo
│   ├── admin_routes.py           # /top-stats, /chart-*, /logs  [Admin only]
│   ├── attendance_routes.py      # /attendance/
│   ├── student_routes.py         # /students/
│   └── auth_routes.py            # /login, /logout, /me
│
├── templates/
│   ├── dashboard.html            # Camera + realtime overlay + điểm danh
│   ├── admin.html                # Dashboard Admin: biểu đồ + nhật ký
│   ├── register.html             # Đăng ký sinh viên nhiều ảnh  [Admin]
│   ├── attendance.html           # Xem điểm danh theo ngày      [Admin]
│   └── history.html              # Lịch sử các ngày trước       [Admin]
│
├── tools/
│   └── enroll_deepfunneled.py    # Batch enroll LFW dataset
│
└── utils/
    ├── auth_utils.py             # @login_required, @admin_required
    └── chart_utils.py            # Sinh biểu đồ confidence (matplotlib)
```

---

## ⚙️ Cài đặt

### Yêu cầu

- Python **3.11+**
- MySQL **8.0+**
- RAM ≥ 8 GB (16 GB nếu enroll toàn bộ LFW)
- NVIDIA GPU + CUDA *(tùy chọn, tăng tốc ~4×)*
- Trình duyệt Chrome / Firefox (WebRTC)

### 1. Clone & tạo môi trường ảo

```bash
git clone https://github.com/<your-username>/FRA-main.git
cd FRA-main

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Cấu hình MySQL

Mở `config.py` và chỉnh thông tin kết nối:

```python
MYSQL_HOST     = "localhost"
MYSQL_PORT     = 3306
MYSQL_USER     = "root"
MYSQL_PASSWORD = "your_password"      # ← thay đổi
MYSQL_DB       = "fra_db"

DATASET_PATH   = r"D:\face_project\dataset\lfw-dataset\lfw-deepfunneled"
```

### 4. Đăng ký dataset LFW *(tùy chọn)*

Script sẽ tự tải mô hình insightface (~170 MB) lần đầu chạy.

```bash
# Đăng ký người có ≥ 2 ảnh (khuyến nghị)
python tools/enroll_deepfunneled.py --min-images 2 --save-npz

# Chỉ đăng ký 100 người để test nhanh
python tools/enroll_deepfunneled.py --limit 100
```

### 5. Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt: **http://localhost:5000**

---

## 🚀 Sử dụng

### Tài khoản mặc định

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Admin |

> ⚠️ **Đổi mật khẩu** ngay sau lần đăng nhập đầu tiên tại *Admin → Đổi mật khẩu*.

### Luồng sử dụng cơ bản

```
1. Register  → Chụp 3–5 ảnh → Đăng ký sinh viên
2. Dashboard → Start camera → Hệ thống tự nhận diện & điểm danh
3. Attendance → Xem danh sách điểm danh theo ngày
4. Admin     → Xem biểu đồ phân tích + nhật ký toàn bộ lần quét
```

---

## 🔌 API

| Method | Endpoint | Auth | Mô tả |
|---|---|---|---|
| POST | `/api/camera/recognize` | — | Nhận diện khuôn mặt từ frame base64 |
| POST | `/api/camera/enroll-multi` | Admin | Đăng ký sinh viên (nhiều ảnh) |
| POST | `/api/camera/enroll-add-photo` | Admin | Thêm ảnh cho sinh viên đã đăng ký |
| GET | `/api/attendance/` | Admin | Danh sách điểm danh theo ngày |
| GET | `/api/attendance/dates` | Admin | Danh sách ngày có dữ liệu |
| GET | `/api/admin/top-stats` | Admin | 4 chỉ số tổng quan |
| GET | `/api/admin/chart-attendance` | Admin | Biểu đồ điểm danh thành công |
| GET | `/api/admin/chart-scans` | Admin | Biểu đồ tất cả lần quét |
| GET | `/api/admin/logs` | Admin | Nhật ký quét (có bộ lọc) |
| POST | `/api/auth/login` | — | Đăng nhập |
| POST | `/api/auth/logout` | — | Đăng xuất |

### Ví dụ gọi API nhận diện

```bash
curl -X POST http://localhost:5000/api/camera/recognize \
  -H "Content-Type: application/json" \
  -d '{"frame": "<base64_jpeg>"}'
```

```json
{
  "faces": [
    {
      "name": "Nguyen Van A",
      "confidence": 0.823,
      "status": "confident",
      "marked": true,
      "bbox": [120.5, 80.2, 280.3, 310.7],
      "det_score": 0.97
    }
  ]
}
```

---

## 🗄️ Cơ sở dữ liệu

```sql
students             -- Embedding trung bình mỗi người (JSON 512 chiều)
student_embeddings   -- Từng embedding riêng lẻ (audit trail)
attendance           -- 1 bản ghi / người / ngày (confidence ≥ 70%)
recognition_logs     -- MỌI lần quét: confident | uncertain | unknown
users                -- Tài khoản admin
```

**Tại sao tách `students` và `student_embeddings`?**
- `students.embedding` = vector trung bình, dùng cho gallery lookup realtime
- `student_embeddings` = từng ảnh riêng → thêm ảnh mới sẽ tính lại trung bình mà không mất dữ liệu cũ

**Tại sao phân biệt `manual` và `lfw`?**
- `manual`: sinh viên thực, chụp qua camera UI
- `lfw`: dữ liệu test từ dataset LFW-deepfunneled

---

## 📊 Hiệu suất thực nghiệm

| Chỉ số | Giá trị |
|---|---|
| Confidence trung bình (lần đầu/người/ngày) | 77–80% |
| Tỉ lệ Confident (≥ 0.70) | > 70% |
| Tốc độ xử lý (CPU) | ~800 ms/frame |
| Tốc độ xử lý (GPU CUDA) | ~200 ms/frame |
| Khoảng cách tối ưu | 0.5 – 1.5 m |
| Số sinh viên đã thử nghiệm | 1.682 (LFW) |

---

## 🛠️ Công nghệ sử dụng

| Layer | Công nghệ |
|---|---|
| Face Detection | [SCRFD](https://github.com/deepinsight/insightface) (insightface buffalo_s) |
| Face Embedding | ArcFace / ResNet-50 (512-dim, L2-normalised) |
| Backend | Python 3.11, Flask 3.0, Waitress WSGI |
| Database | MySQL 8.0, PyMySQL |
| Frontend | Vanilla JS, HTML5 Canvas, WebRTC |
| Charts | Chart.js 4.4 |
| Dataset | LFW-deepfunneled |

---

## 📝 License

MIT © 2024 FRA Project
