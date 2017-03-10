import sys
import tensorflow as tf
from train import Train
from eval import Eval
from test import Pred



def main():
    # Parameters
    # ==================================================

    tf.flags.DEFINE_string("mode", "train", "train, eval, pred")
    # Data loading params
    tf.flags.DEFINE_string("data_file", "query_app.txt", "the data file for model")
    tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_string("checkpoint_dir", "", "dir of the model to be restored")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    if FLAGS.mode == 'train':
        if not FLAGS.data_file:
            print('train_data is required to be provided to train model')
            sys.exit(-1)
        train = Train(FLAGS)
        train.train()
    elif FLAGS.mode == 'eval':
        if not FLAGS.data_file:
            print('valid_data is required to be provided to valid model')
            sys.exit(-1)
        eval = Eval(FLAGS)
        eval.load_data(FLAGS.data_file)
        eval.eval()
    elif FLAGS.mode == 'pred':

        if not FLAGS.data_file:
            print('valid_data is required to be provided to valid model')
            sys.exit(-1)
        pred = Pred(FLAGS)
        for line in sys.stdin:
            line = line[:-1]
            try:
                print(pred.pred(line))
            except ValueError as v:
                print(v)
                continue
    else:
        raise ValueError('Mode not recognized: ' + FLAGS.mode)


if __name__ == '__main__':
    main()