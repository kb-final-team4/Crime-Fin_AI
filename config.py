class Config:

    def __init__(self, model_fn, gpu_id, batch_size, lines, pretrained_model_name):
        # model full name. �� ���� ���
        self.model_fn = model_fn
        # cuda ��� ��, gpu id
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        # �з��ϰ��� �ϴ� text��
        self.lines = lines
        # probability ���� �� ���� ����� ������
        self.top_k = 1
        self.pretrained_model_name = pretrained_model_name
