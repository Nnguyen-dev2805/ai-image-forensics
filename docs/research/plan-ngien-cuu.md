Các trường hợp cần được kiểm tra:

Experiment

Mục đích

Base MLLM

Kiểm khả năng nhìn, classifier và sinh explain có hiệu quả không

Stage1 + LLM

Kiểm tra CLIP/forensic kết hợp với LLM có sinh câu trả lời tốt không

Stage2 + LLM

Kiểm tra train trên token space + LLM có sinh ra câu trả lời tốt không

Stage 2 + Stage 3

Kiểm tra xem train và finetune có đạt chất lượng tốt không

Stage1+Stage2
+Prompt

Kiểm tra chất lượng classifier và chất lượng answer và chuyển đổi space.

Stage 3

Kiểm tra chất lượng riêng MLLM sau khi fine tuning dựa trên dataset.

Stage 1--> 2--> 3

Kiểm tra chất lượng classifier và answer quality khi kết hợp chung
pipeline

(Stage1+2) +
Stage3

Kiểm tra chất lượng classifier và answer quality khi được kết độc lập .
Bởi vì Stage1 + Stage2 học technique, còn stage học semantic. Chúng
có thể không cùng envidence/supervision nếu đi chung 1 flow

1. Mục tiêu nghiên cứu

Dự án hướng tới xây dựng hệ thống phát hiện ảnh thật/ ảnh do AI tạo ra và có khả năng giải
thích được. Hệ thống không chỉ dự đoán nhãn real/fake, mà còn cần đưa lời giải thích bằng
chứng thị giác/forensisc.
Ý tưởng chính: Một hệ thống explainable image detection nên tách thành ba năng lực: forensic

perception, token-space alignment và explanation reasoning. Việc tách giúp model học tín hiệu
ảnh giả tốt hơn, chuyển tín hiệu đó sang không gian MLLM, rồi mới sinh lời giải thích bằng
ngôn ngữ tự nhiên
Các câu hỏi cần được chứng minh:

MLLM gốc có yếu ở fake image detection và fonrensic không reasoning không?

Visual expert ở stage1 có phân loại tốt và generalize tốt hơn CLIP/MLLM thuần không?

Stage2 có thực sự chuyển được fused forensic embedding sang token space của MLLM
không?

Stage3 có cải thiện explaination quality mà không làm giảm classification accuracy không?

Full pipeline có tốt hơn các base line đơn giản như Base MLLM , Stage3 only hoặc Stage 1 +
prompt không?

2. Nguyên tắc triển khai

Không chạy tất cả experiment cùng lúc. Mỗi experiment trả lời một câu hỏi cụ thể. Nếu một
stage thất bại , cần dừng lại nghiên cứu trước khi tiếp tục với các pharse tiếp theo.
Thứ tự ưu tiên:

1. Chuẩn hóa dataset và evaluation protocol

2. Chạy MLLM baseline

3. Train và ablation Stage1

4. Kiểm tra Stage 1 + MLLM/prompt

5. Train Stage 2 alignment

6. Kiểm tra Stage 1 + Stage 2 + prompt

7. Kiểm tra Stage 3 only

8. Train full pipeline Stage 1 --> Stage2 --> Stage3

9. So sánh với biến thể (Stage 1 + Stage2) + Stage 3

3. Phase 0: Chuẩn hóa dataset và evaluation

Mục tiêu: Tạo nền tảng dữ liệu và metric thống nhất để mọi experiment có thể so sánh công
bằng

Dataset train

Stage1 + Stage 2: Tiny GenImage TheKernel01/Tiny-GenImage · Datasets at Hugging Face
Stage3: Holmes zzy0123/AIGI-Holmes-Dataset · Datasets at Hugging Face

Split evaluation

Cần tách rõ:

in-domain: train/test cùng dataset hoặc cùng genertor distribution

cross-generator: test trên genertor chưa thấy khi train'

cross-dataset: train trên dataset này , test trên dataset khác

robustness: test sau JPEG compression , resize, crop, blur nếu có thể

Metric

Classification:

Accuracy

F1-score

AUROC

EER nếu cần

Explanation:

human/GPT-based explanation score

Evidence consistency

Faithfulness score

Error type analysis

Robustness

Cách đánh giá trong bài báo Seeing before Reasoning

3.1. Detection / generalization

Để kiểm tra model có nhận diện ảnh AI, đặc biệt generator unseen, họ dùng nhiều benchmark:

Benchmark

Mục đích

GenImage

Benchmeark AIGI detection tiêu chuẩn

GenImage++ Generator mới và khó hơn

AIGI-Holmes

AIGI detection

WildRF

Ảnh trong điều kiện thực tế , social media

AIGI-Bench

Robustness trong real-world condition

3.2. Explanation - ExplainFake-Bench

Đây là benchmark quan trọng nhất nếu quan tâm đến chất lượng user
Tác giả tự xây: ExplainFake-Bench và sử dụng LLM-as-Judge(GPT -4o) để làm evaluator

Mỗi answer được đánh giá theo 5 dimensions:

Correctness: Final Real/Fake có đúng GT

Specificity : Có đưa evidence cụ thể không

Logical Consistency: Reasoning có ý nghĩa không

Factual Accuracy: Explanation có khớp visual content

instruction Following: Có tuân thủ yêu cầu và format không

Đặc biệt nếu Final Real/Fake judment sai, score sẽ bị significant penalty. Đây là điểm cực kì hay

vì model không thể viết một explanation nghe cực kỳ thuyết phục nhưng kết luận sai mà vẫn
được điểm cao được

ExplainFake-Bench image
        +
Random authenticity instruction
        │
        ▼
   Model được test
        │
        ▼
Generated response
        │
        ├──────── Image
        ├──────── Ground-truth label
        │
        ▼
      GPT-4o
   LLM-as-a-Judge
        │
        ├── Correctness
        ├── Specificity

        ├── Logical Consistency

        ├── Factual Accuracy

        └── Instruction Following

3. General capability preservation

Paper còn kiểm tra forensic training có làm MLLM mất các khả năng vốn có hay không. Đây là
phần general multimodal capability , tách khỏi fake-image detection

Mục tiêu là kiểm tra: Forensic specialization có làm general capability giảm?
Paper so sánh các model forensic với pretrained/general MLLM trên các benchmark mutimodal

tổng quát.

4. Phase 1: Base MLLM baseline

Paper

Base MLLM / backbone
chính

Vai trò

AIGI-Holmes (ICCV
2025)

LLaVA-OneVision-7B

Qwen2.5-VL-7B

Seeing Before
Reasoning /
Forensic-Chat

ForenX

Base MLLM để nhận visual tokens +
instruction và sinh
detection/explanation

Base MLLM được fine-tune theo
framework hai stage

MLLM backbone được
paper cấu hình cho
forensic reasoning

Cần bám đúng bản ForenX để xác
nhận model/version cụ thể

FakeVLM

LLaVA-family VLM

Specialized fake-image
reasoning/detection

Mục đích : Kiểm tra khae năng gốc của MLLM trong việc:

nhìn ảnh và phân loại real/fake

sinh explanation

phát sinh forensic cues

Thiết lập: Không train model. Chỉ dùng MLLM có sẵn với prompt có định

Prompt mẫu:

Is this image real or AI-generated?
Answer with one label: real or fake.
Then briefly explain the visual evidence.

Kết quả cần lưu:

Label prediction

Confidence nếu model có sinh

Explanation

Lỗi thường gặp

Câu hỏi cần trả lời:

MLLM có phân loại tốt không?

Explanation có cụ thể hay nói chung chung?

Model dựa vào semantic hay forensic cues ?

Model có hallucinate bằng chứng không?

5. Phase 2: Stage 1 - Classification Expert training

Mục đích: Train visual expert chuyên phân biệt ảnh thật và ảnh giả bằng hai loại tín hiệu:

semantic cues từ CLIP

low-level forensic từ NPR-Resnet

Kiến trúc:

Ảnh đầu vào đi qua hai nhánh:

CLIP Vision encoder : lấy semantic/general visual representation

NPR + ResNet: lấy forensic representation như texture , noise, edge artifact, over smoothing

Sau đó được fusion feature để train classifier real/fake

Model

Mục đích

CLIP-only classifier Kiểm tra semantic branch có đủ mạnh không

NPR-ResNet-only

Kiểm tra forensic branch có đủ mạnh không

CLIP + NPR fusion

Kiểm tra fusion có tốt hơn từng nhánh riêng không

Evaluation

in-domain test

Cross - generator test

Cross - dataset test

JPEG compression robustness

Resize/crop robustness

Câu hỏi cần trả lời:

Fusion có tốt hơn CLIP-only và NPR-only không

NPR branch có giúp unseen generator không?

CLIP branch có giúp giảm false positive trên ảnh real không ?

Model có học shortcut từ dataset không?

Tiêu chí và kết quả mong muốn:

fusion tốt hơn hoặc ít nhất ổn hơn từng branch riêng

performance trên unseen generator không sụp mạnh

robustness không quá kém khi ảnh bị nén hoặc resize

6. Phase 3: Stage 1 + LLM

Mục đích: Kiểm tra output từ Stage1 có giúp LLM sinh câu trả lời tốt hơn Base MLLM không.
LLM nhận thông tin từ Stage 1 sau đó đưa ra dự đoán và bằng chứng

Câu hỏi cần trả lời:

LLM có viết explanation tốt hơn Base MLLM không?

Explanation có bám vào output của Stage1 không?

Nếu Stage 1 sai, LLM có bị kéo sai theo không?

Prompt-based có đủ tốt để không cần Stage 2 không?

7. Phase 4: Stage2 - TokenSpace

Mục đích: Train projector để chuyển fused visual embedding từ Stage 1 sang hidden/token
space mà MLLM đóng băng có thể sử dụng

Thiết lập:

Freeze Stage 1

Freeze MLLM

Chỉ train projection

Objective ban đầu: MLLM predict đúng token real/fake

Input:

Fused embedding từ Stage 1

prompt ngắn yêu cầu trả lời real/fake
Output:

token real hoặc fake

Evaluation

Real/Fake token accuracy

So sánh với Stage 1 classifier gốc

So sánh với Stage 1 + Prompt

Câu hỏi cần được trả lời:

Projector có giữ được năng lực phân loại từ Stage 1 không?

MLLM đóng băng có đọc được visual tokens không?

Stage 2 có cải thiện so với prompt-based Stage 1 + LLM không?

Alignment có giúp explanation về sau không?

Tiêu chí và kết quả mong muốn:

token accuracy gần với Stage 1 classifier accuracy

performance không giảm mạnh trên cross-generator

visual tokens giúp MLLM trả lời ổn định hơn prompt - only

8. Phase 5: Stage 1 + 2 + Prompt

Mục đích : Kiểm tra sau khi alignment, chỉ cần prompt nhẹ thì MLLM có thể trả lời và giải thích

tốt chưa

Thiết lập:

Stage 1 frozen

Stage 2 projector frozen

MLLM chưa fine tune explanation

Prompt yêu cầu output real/fake và explanation ngắn

Câu hỏi cần đươc trả lời:

Label accuracy có tốt không?

Explanation có tốt hơn Base MLLM không?

Nếu label đúng nhưng explanation tệ, Stage 3 là cần thiết

Nếu label tệ, Stage 2 alignment chưa ổn

9. Phase 6: Stage 3 - explanation reasoning training

Mục đích: Kiểm tra khả năng MLLM sinh explanation sau fine-tune trên explanation dataset

(Holmes)

Thiết lập:

Fine - tune trên MLLM hoặc LoRA trên dataset có label + explanation

Có thể chưa cần nối với Stage 1/2

Input là ảnh + instruction

Output là label + explanation

Câu hỏi cần được trả lời:

Stage 3-only có cải thiện explanation fluency không?

Label accuracy có tăng hay giảm

Model có học temple explanation không

Explanation có faithful với ảnh không
Rủi ro: Stage3-only có thể viết rất hay nhưng không nhìn vào forensic evidence. Vì vậy
không được chỉ đánh giá bằng độ mượt của câu trả lời

Tiêu chí và đánh giá:

classification metrics

explanation quality score

qualitative error analysis

so sánh với Base MLLM

10. Phase 7: Full sequential pipeline

Stage 1 --> Stage 2 --> Stage 3

Mục đích: Kiểm tra chất lượng full pipeline khi train đúng thứ tự:

Train Stage 1 visual expert

Freeze Stage 1

Train Stage 2 projector

Freeze Stage 1 + Stage 2

Train Stage 3 explanation/reasoning

Câu hỏi cần được trả lời:

Full pipeline có tốt hơn Base MLLM không ?

Full pipeline có tốt hơn Stage 3 - only không ?

Full pipeline có tốt hơn Stage 1 + prompt không?

Stage 3 có làm giảm classification accuracy không?

Explanation có bám vào evidence từ Stage 1/2 không?

Tiêu chí và kết quả mong muốn:

classification accuracy tốt hơn Base MLLM và Stage 3 - only

cross-generator performance tốt

explanation cụ thể hơn Base MLLM

faithful tốt hơn Stage 3-only

không làm mất năng lực phân lại của stage 1

11. Phase: Biến thể pipeline

Lý do: Do dataset dùng cho stage 3 mang ý nghĩa nhiều về semantic (thị giác), không thiên về
kỹ thuật nên có thể bất đồng bộ hoặc xung đột với các evidence trong Stage1+2

Mục đích: Kiểm tra giả thuyết forensic alignment và explanation reasoning có thể nên được học
độc lập

Thiết lập: Train riêng

Module A: Stage 1 + Stage 2 để học forensic perception và token-space alignment

Module B: Stage 3 để học explanation generation

So sánh cần có:

Setup

Ý nghĩa

Stage 1 -> Stage 2 -> Stage 3

Pipeline tuần tự

(Stage 1 + Stage 2) + Stage 3

Forensic alignment và explanation học độc lập

Stage 3-only

Chỉ fine-tune MLLM

Stage 1 + prompt

Classifier-assisted explanation

Câu hỏi cần trả lời:

Independent combination có ổn định hơn sequential training không?

Explanation có ít hallucinate hơn không?

Classification accuracy có giữ tốt hơn không?

Hai loại supervision forensic và semantic có xung đột không?

