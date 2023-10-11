import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, 'module/text_classification'))
from module.text_classification.inference import TextClassifier


config = {
    'checkpoint_lst': [os.path.join(root, 'model/rhetoric_model/Analogy_PretrainedBert_1e-05_16_0.5.pt')],
    'use_bert': True,
    'embd_path': os.path.join(root, 'model/word_embedding/tencent_small'),
    'model_config_lst': [{
        'is_state': False,
        'model_name': 'bert',
        'pretrained_model_path': os.path.join(root, 'model/bert_chinese_wwm_ext_pytorch')}]
}
def predict_sentences(sent_list, config):
    pretrained_model_path = config['model_config_lst'][0]['pretrained_model_path']
    model = TextClassifier(config['embd_path'], config['checkpoint_lst'], config['model_config_lst'],
                           pretrained_model_path)
    max_seq_len =config['max_seq_len'] if 'max_seq_len' in config else 80
    need_mask = config['need_mask'] if 'need_mask' in config else False
    pred_list, proba_list = model.predict_all_mask(sent_list, max_seq_len=max_seq_len, max_batch_size=20,
                                                   need_mask=need_mask)
    pos_sent_list = [sent_list[i] for i in range(len(pred_list)) if pred_list[i] == 1]
    print(pred_list, proba_list)
    print("比喻数目", len(pos_sent_list))

if __name__ == "__main__":
    sent_list = [
        '他的眼睛像一汪清泉，清澈见底。',
        '草地上 传来他们一阵阵的欢呼声，如银铃般清脆,如驼铃般悠远，与这美好的 舂色融为了一体。',
        '小睿举着一只美丽的燕子风筝，小明在前面像一名百米冲刺的运动员，低着头只顾拼命地拉着线奔跑。'
    ]
    
    # 调用函数：
    predict_sentences(sent_list,config)
