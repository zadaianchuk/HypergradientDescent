# -*- coding: utf-8 -*-
"""
Run HyperGradientDescentOptimizer on a test problem.

Runs HyperGradientDescentOptimizer
on a test problem from the tfobs package.
  - All necessary parameters and options are passed via the command line.
  - At regularly-spaced checkpoints during training, loss and accuracy are
    evaluated on training and test data.
  - All results are saved to pickle files and logged with tensorflow
    summaries. (The latter can be switched off.) We also log all tensorflow
    summaries created elsewhere, as long as they are tagged with the key
    "per_iteration".

Usage:

    python run_HGD.py <dataset>.<problem> --all_the_command_line_arguments

Execute python run_HGD.py --help to see a description for the command line
arguments. The command line arguments separate into the following categories:
  - arguments needed to set up the problem: "test_problem" (the first (and only)
    positional argument), "batch_size", "data_dir", "random_seed"
  - The basic mechanis: "num_steps", "checkpoint_interval", "eval_size"
  - Learning rate: "lr" for constant learning rates; "lr_sched_steps" and
    "lr_sched_vals" for step-wise schedules; other options might be added later
  - Optimizer parameters other than learning rates: only the momentum parameter
    "mu" in this case, but there might be more for other optimizers
  - Logging options: "train_log_interval", "saveto", "logdir", "nologs"
"""

import tensorflow as tf
import pickle
import argparse
import importlib
import time
import os
import tfobs
import Adam_HD_optimizer as opts
reload(opts)

# ------- Parse Command Line Arguments ----------------------------------------
parser = argparse.ArgumentParser(description="Run HGDOptimizer on a tfobs "
    "test problem.")
parser.add_argument("test_problem",
    help="Name of the test_problem (e.g. 'cifar10.cifar10_3c3d'")
parser.add_argument("--data_dir",
    help="Path to the base data dir. If not set, tfobs uses its default.")
parser.add_argument("--bs", "--batch_size", required=True, type=int,
    help="The batch size (positive integer).")

# Optimizer hyperparams other than learning rate
parser.add_argument("--alpha_0",  type=float, default=0.0001,
    help="Initial value of step size")
parser.add_argument("--type", default = "global",
    help="Set to 'global' if you want to optimize one dimentional learning rate. Set to 'local' if you want to find per coordinate optimal learning rate" )

parser.add_argument("--beta", type=float, default=10**(-8),
    help="Constant learning rate (positive float) to use. To set a learning "
    "rate *schedule*, do *not* set '--lr' and use '--lr_sched_steps' "
    "and '--lr_sched_values' instead.")

# Number of steps and checkpoint interval
parser.add_argument("-N", "--num_steps", required=True, type=int,
                    help="Total number of training steps.")
parser.add_argument("-C", "--checkpoint_interval", required=True, type=int,
    help="Interval of training steps at which to evaluate on the test set and "
    "on a larger chunk of the training set.")
parser.add_argument("--eval_size", type=int, default=10000,
    help="Number of data points used for evaluation at checkpoints. This should "
    "usually be the test set size (the default is 10000, the test set size of "
    "MNIST, CIFAR). We evaluate on floor(eval_size/batch_size) batches. "
    "The number is the same for test and training evaluation.")

# Random seed
parser.add_argument("-r", "--random_seed", type=int, default=42,
    help="An integer to set as tensorflow's random seed.")

# Logging
parser.add_argument("--train_log_interval", type=int, default=10,
    help="The interval of steps at which the mini-batch training loss is "
    "logged. Set to 1 to log every training step. Default is 10.")
parser.add_argument("--print_train_iter", action="store_const",
    const=True, default=False,
    help="Add this flag to print training loss to stdout at each logged "
    "training step.")
parser.add_argument("--saveto", default = "results",
    help="Folder for saving the results file. If not specified, defaults to "
    "'results/<test_problem>.' The directory will be created if it does not "
    "already exist.")
parser.add_argument("--nologs", action="store_const", const=True, default=False,
    help="Add this flag to switch off tensorflow logging.")

args = parser.parse_args()
# -----------------------------------------------------------------------------

# Create an identifier for this experiment
name = args.test_problem.split(".")[-1]
name += "__HGD"
name += "__bs_" + str(args.bs)
name += "__alpha_0_" + str(args.alpha_0)
name += "__type_"+args.type
name += "__beta_" + str(args.beta)
name += "__N_" + str(args.num_steps)
name += "__seed_" + str(args.random_seed)

# Set the log/results directory (create if it does not exist)
logdir = os.path.join(args.saveto, args.test_problem.split(".")[-1])
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Set the data dir
if args.data_dir is not None:
  tfobs.dataset_utils.set_data_dir(args.data_dir)

# Number of evaluation iterations
num_eval_iters = args.eval_size//args.bs

# Set up test problem
test_problem = importlib.import_module("tfobs."+args.test_problem)
tf.reset_default_graph()
tf.set_random_seed(args.random_seed) # Set random seed
losses,  accuracy, phase = test_problem.set_up(
    batch_size=args.bs)
loss = tf.reduce_mean(losses)

# If there are terms in the REGULARIZATION_LOSSES collection, add them to the loss
regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
if regularization_losses:
    loss = loss + tf.add_n(regularization_losses)

# Learning rate tensor; constant or schedule
global_step = tf.Variable(0, trainable=False)

# Set up optimizer
opt =  opts.AdamHDOptimizer(alpha_0=args.alpha_0,beta=args.beta, type_of_learning_rate=args.type)
step = opt.minimize(loss, global_step=global_step)

# Lists for tracking stuff
# train_<quantity>[i] is <quantity> after training for train_steps[i] steps
# checkpoint_<quantity>[i] is <quantity> after training for checkpoint_steps[i] steps
train_steps = []
train_losses = []
checkpoint_steps = []
checkpoint_train_losses = []
checkpoint_train_accuracies = []
checkpoint_test_losses = []
checkpoint_test_accuracies = []

# Tensorboard summaries
if not args.nologs:
    train_loss_summary = tf.summary.scalar("training_loss", loss,
                                           collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
    per_iteration_summaries = tf.summary.merge_all(key="per_iteration")
    summary_writer = tf.summary.FileWriter(os.path.join(logdir, "tflogs__"+name))

# ------- start of train looop --------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for n in range(args.num_steps + 1):

    # Evaluate if we hit the checkpoint interval (and in the last step)
    if n % args.checkpoint_interval == 0 or n == args.num_steps:

        print("********************************")
        print("CHECKPOINT (", n, "of", args.num_steps, "steps )")
        checkpoint_steps.append(n)

        # Evaluate on training data
        train_loss_, train_acc_ = 0.0, 0.0
        for _ in range(num_eval_iters):
            l_, a_ = sess.run([loss, accuracy], {phase: "train_eval"})
            train_loss_ += l_
            train_acc_ += a_
        train_loss_ /= float(num_eval_iters)
        train_acc_ /= float(num_eval_iters)

        # Evaluate on test data
        test_loss_, test_acc_ = 0.0, 0.0
        for _ in range(num_eval_iters):
            l_, a_ = sess.run([loss, accuracy], {phase: "test"})
            test_loss_ += l_
            test_acc_ += a_
        test_loss_ /= float(num_eval_iters)
        test_acc_ /= float(num_eval_iters)

        # Append results to lists
        checkpoint_train_losses.append(train_loss_)
        checkpoint_train_accuracies.append(train_acc_)
        checkpoint_test_losses.append(test_loss_)
        checkpoint_test_accuracies.append(test_acc_)

        # Log results to tensorflow summaries
        if not args.nologs:
            summary = tf.Summary()
            summary.value.add(tag="checkpoint_train_loss",
                              simple_value=train_loss_)
            summary.value.add(tag="checkpoint_train_acc",
                              simple_value=train_acc_)
            summary.value.add(tag="checkpoint_test_loss",
                              simple_value=test_loss_)
            summary.value.add(tag="checkpoint_test_acc",
                              simple_value=test_acc_)
            summary_writer.add_summary(summary, n)
            summary_writer.flush()

        print("TRAIN: loss", train_loss_, "acc", train_acc_)
        print("TEST: loss", test_loss_, "acc", test_acc_)
        print("********************************")

        # Break from train loop after the last round of evaluation
        if n == args.num_steps:
            break

    # Training step, with logging if we hit the train_log_interval
    if n % args.train_log_interval != 0:
        _ = sess.run(step, {phase: "train"})
    else:  # if n%args.train_log_interval==0:
        if not args.nologs:
            _, loss_, per_iter_summary_ = sess.run([step, loss, per_iteration_summaries],
                                                   {phase: "train"})
            summary_writer.add_summary(per_iter_summary_, n)
        else:
            _, loss_ = sess.run([step, loss], {phase: "train"})
        train_steps.append(n)
        train_losses.append(loss_)
        if args.print_train_iter:
            print("Step", n, ": loss", loss_)
sess.close()
# ------- end of train looop --------

# Put logged stuff into results dict and save to pickle file
results = {
    "args": args,
    "train_steps": train_steps,
    "train_losses": train_losses,
    "checkpoint_steps": checkpoint_steps,
    "checkpoint_train_losses": checkpoint_train_losses,
    "checkpoint_train_accuracies": checkpoint_train_accuracies,
    "checkpoint_test_losses": checkpoint_test_losses,
    "checkpoint_test_accuracies": checkpoint_test_accuracies
}
with open(os.path.join(logdir, "results__"+name+".pickle"), "w") as f:
    pickle.dump(results, f)
