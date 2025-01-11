class Config:

    def __init__(self, model_fn, gpu_id, batch_size, lines, pretrained_model_name):
        # model full name. 모델 저장 경로
        self.model_fn = model_fn
        # cuda 사용 시, gpu id
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        # 분류하고자 하는 text들
        self.lines = lines
        # probability 상위 몇 개를 출력할 것인지
        self.top_k = 1
        self.pretrained_model_name = pretrained_model_name
