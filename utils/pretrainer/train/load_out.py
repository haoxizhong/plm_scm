from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from pytorch_pretrained_bert.modeling import BertForPreTraining

input_path = "/data/disk2/private/zhx/scm/pretrain/all/pytorch_model.checkpoint2"
output_path = "/data/disk2/private/zhx/scm/pretrain/final/pytorch_model.bin"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def checkpoint(model, output_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = output_path
    params = model_to_save.state_dict()
    try:
        torch.save(params, output_model_file)
    except BaseException:
        logger.warning('WARN: Saving failed... continuing anyway.')


def load_checkpoint(filename):
    logger.info('Loading model %s' % filename)
    saved_params = torch.load(
        filename, map_location=lambda storage, loc: storage
    )
    args = saved_params['args']
    global_step = saved_params['global_step']
    model_dict = saved_params['model_dict']
    optimizer_dict = saved_params['optimizer']
    iter_id = saved_params['iter_id']

    model = BertForPreTraining.from_pretrained(args.bert_model, state_dict=model_dict)
    return model


def main():
    model = load_checkpoint(input_path)
    checkpoint(model, output_path)


if __name__ == "__main__":
    main()
