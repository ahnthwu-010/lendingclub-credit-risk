# LendingClub Credit Risk — Survival, Causal & Policy Optimization

Phân tích rủi ro tín dụng trên **2,250,076 khoản vay** từ LendingClub (2007–2018), kết hợp Competing Risks Survival Analysis, Predictive Modeling và Causal Inference để trả lời câu hỏi kinh doanh: **chính sách xác minh thu nhập/nguồn thu (verification) hiện tại có đang lãng phí tiền không, và nên áp dụng chọn lọc thế nào để tối ưu chi phí?**

**Dashboard trực quan:** [(https://lendingclub-credit-risk-kkujezq9pbydbufk9frqbw.streamlit.app/)]

---

## 1. Bài toán kinh doanh

LendingClub cấp khoản vay dựa trên hồ sơ tự khai, với 3 mức xác minh: `Verified`, `Source Verified`, `Not Verified`. Xác minh tốn chi phí vận hành ($20/khoản vay, giả định trong bài) nhưng lý thuyết sẽ giảm rủi ro vỡ nợ. Phân tích này kiểm chứng bằng dữ liệu thật: **xác minh có thực sự giảm rủi ro không, hay chỉ là phản ứng với rủi ro đã cao sẵn (reverse causation)?**

3 tầng phân tích liên kết:
1. **Survival Analysis** — khoản vay "sống" bao lâu trước khi vỡ nợ hoặc trả hết sớm (Competing Risks)
2. **Predictive Modeling** — dự đoán xác suất vỡ nợ
3. **Causal Inference (X-learner)** — tách bạch tác động thật của xác minh khỏi tương quan → dùng tối ưu chính sách

---

## 2. Dữ liệu

- Nguồn: [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — `accepted_2007_to_2018Q4.csv.gz` (392.6MB, 151 cột)
- Sau làm sạch: **2,250,076 khoản vay** (loại 10,592 dòng lỗi ngày tháng, 0.47%)
- Phân bố outcome: Fully Paid 1,068,731 · Charged Off 268,880 · Censored (đang vay) 912,465

| event_type | Ý nghĩa |
|---|---|
| 0 | Censored — Current/Late/Grace Period |
| 1 | Charged Off (vỡ nợ) — rủi ro chính |
| 2 | Fully Paid — rủi ro cạnh tranh |

---

## 3. Kết quả chính — Business Insights

### 3.1 Ước lượng risk bằng phương pháp sai làm bóp méo con số 40%

Dùng Kaplan-Meier thông thường (coi Fully Paid là censored — sai về mặt lý thuyết) cho ra xác suất vỡ nợ tại tháng 36 là **20.88%**. Dùng đúng Aalen-Johansen (Competing Risks) chỉ ra con số thật là **15.48%** — lệch **5.40 điểm %**, tức phương pháp sai làm phóng đại rủi ro gần **35%** so với thực tế. Đây là bằng chứng cụ thể tại sao chọn đúng phương pháp thống kê quan trọng hơn "chạy được model".

### 3.2 Rủi ro vỡ nợ tăng dốc theo Grade — nhưng không tuyến tính

| Grade | Rủi ro vỡ nợ (36 tháng) | Số lượng |
|---|---|---|
| A | 5.02% | 431,473 |
| B | 10.63% | 660,649 |
| C | 17.05% | 646,907 |
| D | 23.75% | 322,643 |
| E | 30.29% | 134,839 |
| F | 36.84% | 41,530 |
| G | 41.54% | 12,035 |

Grade là yếu tố phân tách rủi ro mạnh nhất trong toàn bộ pipeline (được Cox PH xác nhận: HR Grade G = 5.54 so với Grade A, p<0.005).

### 3.3 Phát hiện quan trọng nhất: Verification tương quan với rủi ro CAO HƠN, không thấp hơn

- Cox PH (Concordance 0.6823): `verification_status_Verified` có HR=1.20, `Source Verified` HR=1.16 — cả hai đều **tăng** hazard vỡ nợ (p<0.005)
- So sánh thô (chưa điều chỉnh): nhóm được xác minh có tỷ lệ default **22.39%**, nhóm không xác minh chỉ **14.82%** — chênh lệch **+7.57 điểm %** theo hướng "xác minh tệ hơn"

→ Đây **không phải** bằng chứng xác minh gây hại — mà là dấu hiệu kinh điển của **reverse causation**: LendingClub có xu hướng yêu cầu xác minh nhiều hơn với khoản vay *đã* có dấu hiệu rủi ro cao (thu nhập cao bất thường, hồ sơ mập mờ...), không phải verification tạo ra rủi ro. Đây chính là lý do bắt buộc phải dùng Causal Inference thay vì đọc trực tiếp số liệu thô.

### 3.4 Causal Inference (X-learner): tách nhân quả thật khỏi tương quan

- Propensity overlap chấp nhận được (treatment mean 0.7255 vs control 0.6298, range chồng lấn tốt) → đủ điều kiện ước lượng CATE tin cậy
- **ATE toàn dataset: -1.45 điểm %** (âm = xác minh không giúp giảm rủi ro trên trung bình toàn bộ, ngược với kỳ vọng ban đầu) — sau khi đã điều chỉnh confounding, hiệu ứng nhân quả thật **nhỏ hơn nhiều** so với chênh lệch thô +7.57 điểm % → phần lớn chênh lệch thô là do reverse causation, không phải do xác minh gây hại
- Chỉ **18.44%** khoản vay có CATE dương (xác minh thực sự giúp giảm rủi ro cho nhóm này)

**CATE trung bình theo Grade — insight cốt lõi cho chính sách:**

| Grade | CATE trung bình | Diễn giải |
|---|---|---|
| A | -0.46% | Xác minh gần như không có tác dụng |
| B | -0.93% | Không có tác dụng |
| C | -2.05% | Không có tác dụng |
| D | -2.66% | Không có tác dụng |
| E | -1.45% | Không có tác dụng |
| F | -1.30% | Không có tác dụng |
| **G** | **+2.27%** | **Xác minh thực sự giúp giảm rủi ro** |

→ Xác minh chỉ có tác dụng nhân quả thật ở nhóm rủi ro cao nhất (Grade G). Với Grade A-F, xác minh đại trà là **lãng phí chi phí vận hành mà không cải thiện rủi ro**.

**CATE theo mức DTI (Debt-to-Income) — kiểm tra thêm yếu tố phân hóa:**

| DTI Quartile | CATE trung bình |
|---|---|
| Q1 (thấp nhất) | -1.37% |
| Q2 | -1.35% |
| Q3 | -1.42% |
| Q4 (cao nhất) | -1.65% |

Chênh lệch giữa các quartile rất nhỏ (~0.3 điểm %) — DTI **không phải yếu tố phân hóa** hiệu quả xác minh, khác hẳn Grade (chênh từ -2.66% đến +2.27%). Kết luận vận hành: chính sách chọn lọc chỉ cần dựa vào Grade, không cần thêm điều kiện DTI, giúp rule đơn giản hơn khi triển khai thực tế.

### 3.5 LGD (Loss Given Default) — tương đối đồng đều, Grade G nặng nhất

| Grade | LGD trung bình |
|---|---|
| A | 40.35% |
| B | 39.58% |
| C | 40.71% |
| D | 40.88% |
| E | 40.16% |
| F | 40.48% |
| **G** | **43.46%** |

### 3.6 Kết luận tối ưu chính sách — con số tài chính cụ thể

Trên mẫu 300,000 khoản vay:

| Chính sách | Chi phí | Net Value |
|---|---|---|
| Xác minh **đại trà** (hiện tại) | $6,000,000 | **-$20,044,319** (lỗ ròng) |
| Xác minh **chọn lọc** (chỉ khi Expected Value > 0) | $917,640 (**giảm 84.7%**) | **+$9,384,462** |

**Cải thiện: +$29,428,781** trên mẫu 300K khoản vay — chỉ **15.29%** khoản vay (45,882 khoản) thực sự đáng xác minh, tập trung mạnh ở Grade rủi ro cao:

| Grade | % khoản vay đáng xác minh |
|---|---|
| G | 61.4% |
| F | 33.5% |
| E | 26.1% |
| A | 15.3% |
| D | 14.1% |
| B | 13.4% |
| C | 12.5% |

**Sensitivity Analysis** — kết luận vẫn đúng ngay cả khi giảm mạnh độ tin tưởng vào CATE:

| Mức tin tưởng CATE | Số khoản đáng verify | % tổng | Net Value |
|---|---|---|---|
| 100% | 45,882 | 15.3% | $9,384,462 |
| 75% | 43,881 | 14.6% | $6,814,006 |
| 50% | 40,549 | 13.5% | $4,261,631 |
| 25% | 32,573 | 10.9% | $1,766,499 |
| 10% | 17,727 | 5.9% | $416,484 |

Ngay cả khi chỉ tin **10%** vào ước lượng CATE, chính sách chọn lọc vẫn dương — kết luận **rất bền vững**, không phải kết quả may rủi của model.

**Kịch bản triển khai đơn giản nhất (dễ giải thích cho non-technical):** chỉ xác minh Grade F & G — 3,662 khoản vay, chi phí $73,240, net value **+$1,463,976**. Không cần model CATE phức tạp, chỉ cần 1 rule đơn giản theo Grade vẫn có lời.

### 3.7 Giới hạn thực tế của model dự đoán — không nên tự động hóa hoàn toàn

Model XGBoost bắt được **67% khoản vay sẽ vỡ nợ thật** (recall) nhưng chỉ **32% khoản bị gắn cờ rủi ro là đúng** (precision) — cứ 3 khoản vay bị model đánh dấu rủi ro cao, chỉ 1 khoản thật sự vỡ nợ.

**Ý nghĩa vận hành:** không nên dùng model này để **tự động từ chối** khoản vay — sẽ từ chối oan rất nhiều khách hàng tốt (false positive cao). Nên dùng như **lớp chấm điểm ưu tiên** để đội thẩm định xem xét thủ công những khoản có điểm rủi ro cao, kết hợp với rule chọn lọc theo Grade ở mục 3.6 để quyết định có xác minh hay không.

### Khuyến nghị kinh doanh

**Chuyển từ xác minh đại trà sang xác minh chọn lọc, ưu tiên Grade F/G.** Chính sách hiện tại (xác minh gần 70% khoản vay) đang lỗ ròng ~$20M/300K khoản vay vì áp dụng tràn lan cho cả nhóm rủi ro thấp (A-C) — nơi xác minh không có tác dụng nhân quả thật. Chỉ cần giới hạn xác minh vào Grade F-G (đơn giản nhất) hoặc dùng model CATE đầy đủ (tối ưu nhất, $9.4M net value) đều mang lại cải thiện tài chính lớn mà không cần thêm chi phí công nghệ đáng kể. Model dự đoán vỡ nợ nên đóng vai trò hỗ trợ ưu tiên xem xét, không thay thế quyết định thủ công do precision còn thấp (32%).

---

## 4. Phương pháp kỹ thuật

- **Competing Risks**: Kaplan-Meier (baseline sai) vs Aalen-Johansen CIF (đúng), Cause-Specific Cox PH cho 2 outcome (loại `int_rate` do đa cộng tuyến với `grade`)
- **Predictive**: XGBoost trên 1,337,611 khoản vay "mature" (default rate 20.10%), AUC 0.7166, PR-AUC 0.3877; feature quan trọng nhất: `int_rate` (23.8%), `term_60_months` (8.8%), `grade_B` (6.5%)
- **Causal**: Treatment = `verification_status != 'Not Verified'` (69.79% khoản vay), X-learner (2 outcome model + 2 tau model theo propensity), kiểm tra overlap trước khi ước lượng
- **Optimization**: LGD tính từ dữ liệu Charged Off thật, Expected Value = CATE × Potential Loss − Chi phí xác minh ($20/khoản)

---

## 5. Cấu trúc project

├── 01_data_prep_survival.ipynb # Load, clean, Competing Risks, Cox PH
├── 02_predictive_causal.ipynb # XGBoost, X-learner CATE, Heterogeneity
├── 03_optimization.ipynb # LGD, Policy Optimization, Sensitivity
├── data/
│ ├── lendingclub_with_cate.csv
│ ├── lendingclub_final_optimization.csv
│ ├── dashboard_data/
│ ├── lgd_by_grade.csv
│ ├── model_features.csv
│ └── xgb_default_model.json
└── app.py

## 6. Chạy lại

```bash
pip install pandas numpy lifelines xgboost scikit-learn matplotlib streamlit
```

Data thô (392.6MB) tải từ [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club), không đính kèm trong repo do giới hạn dung lượng GitHub.

## 7. Hạn chế

- Model Cox và XGBoost train trên mẫu 200K–300K (không phải full 2.25M) vì giới hạn RAM local — kết quả ổn định nhưng nên lưu ý khi diễn giải
- Treatment không randomized — dù đã propensity-adjust, vẫn có rủi ro confounding chưa quan sát (VD: lý do cụ thể khoản vay bị gắn cờ verification không có trong dữ liệu)
- Concordance Cox Fully Paid chỉ 0.5942 — model yếu hơn đáng kể so với model Charged Off (0.6823), nên diễn giải hazard ratio của outcome này thận trọng hơn
- Precision model XGBoost chỉ 32% — không phù hợp để tự động hóa quyết định từ chối khoản vay
- Chi phí xác minh $20/khoản là giả định, chưa xác nhận với số liệu vận hành thật của LendingClub

