from __future__ import division

import tensorflow as tf

class AdamHDOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, alpha_0, beta =10**(-7), name="HGD", mu=0.99, eps = 10**(-8),type_of_learning_rate ="global"):
        super(AdamHDOptimizer, self).__init__(beta, name=name)

        self._mu = mu
        self._alpha_0 = alpha_0
        self._beta = beta
        self._eps = eps
        self._type = type_of_learning_rate


    def minimize(self, loss, global_step):

        # Algo params as constant tensors
        mu = tf.convert_to_tensor(self._mu, dtype=tf.float32)
        alpha_0=tf.convert_to_tensor(self._alpha_0, dtype=tf.float32)
        beta=tf.convert_to_tensor(self._beta, dtype=tf.float32)
        eps = tf.convert_to_tensor(self._eps, dtype=tf.float32)

        var_list = tf.trainable_variables()


        # Create and retrieve slot variables for delta , old_grad values
        # and old_dir (values of gradient changes)

        ds = [self._get_or_make_slot(var,
                  tf.constant(0.0, tf.float32, var.get_shape()), "direction", "direction")
                  for var in var_list]
        if self._type == "global":
            alpha = self._get_or_make_slot(alpha_0, alpha_0, "learning_rate", "learning_rate")
        else:
            alphas = [self._get_or_make_slot(var,
                      tf.constant(self._alpha_0, tf.float32, var.get_shape()), "learning_rates", "learning_rates")
                      for var in var_list]

        # moving average estimation
        ms = [self._get_or_make_slot(var,
            tf.constant(0.0, tf.float32, var.get_shape()), "m", "m")
            for var in var_list]
        vs = [self._get_or_make_slot(var,
            tf.constant(0.0, tf.float32, var.get_shape()), "v", "v")
            for var in var_list]
        # power of mu for bias-corrected first and second moment estimate
        mu_power = tf.get_variable("mu_power", shape=(), dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(1.0))


        grads = tf.gradients(loss, var_list)
        grads_squared = [tf.square(g) for g in grads]
        if self._type == "global":
            hypergrad = sum([tf.reduce_sum(tf.multiply(d,g)) for d,g in zip(ds, grads)])
            alphas_update = [alpha.assign(alpha-beta*hypergrad)]
        else:
            hypergrads = [tf.multiply(d,g) for d,g in zip(ds, grads)]
            alphas_update = [alpha.assign(alpha-beta*hypergrad) for alpha,hypergrad in zip(alphas,hypergrads)]
        # alpha=tf.Print(alpha,[alpha], message="This is alpha: ")
        m_updates = [m.assign(mu*m + (1.0-mu)*g) for m, g in zip(ms, grads)] #new means
        v_updates = [v.assign(mu*v + (1.0-mu)*g2) for v, g2 in zip(vs, grads_squared)]
        mu_power_update = [tf.assign(mu_power,tf.multiply(mu_power,mu))]

        #calculate probability of sign switch
        with tf.control_dependencies(v_updates+m_updates+alphas_update+mu_power_update):
            #bais_correction
            # mu_power = tf.Print (mu_power,[mu_power])
            ms_hat = [tf.divide(m,tf.constant(1.0) - mu_power) for m in ms]
            vs_hat = [tf.divide(v,tf.constant(1.0) - mu_power) for v in vs]
            ms_squared = [tf.square(m) for m in ms]

            rs = [tf.maximum(v-m2,tf.zeros_like(v)) for v, m2 in zip(vs_hat, ms_squared)] #new varience
            # probability of sign switch (with equal variance assumption)
            # x=tf.Print(sum([tf.reduce_sum(m**2) for m in ms_hat]),[sum([tf.reduce_sum(m**2) for m in ms_hat])], message="This is m_hat: ")
            # y=tf.Print(sum([tf.reduce_sum(v**2) for v in ms]),[sum([tf.reduce_sum(v**2) for v in ms])], message="This is m: ")
            snrs = [tf.divide(m, tf.sqrt(r) + self._eps) for m, r in zip(ms_hat, rs)]
            # summary histogram SNR
            abs_snrs =[tf.abs(snr) for snr in snrs]

            # w update directions
            ds_updates=[d.assign(-snr) for (snr,d) in zip(snrs,ds)]
        with tf.control_dependencies(ds_updates):
                if self._type == "global":
                    dirs = [alpha*d for  d in ds]
                    alpha_norm = alpha
                else:
                    dirs = [alpha*d for  d, alpha in zip(ds,alphas)]
                    alpha_norm = sum([tf.reduce_mean(alpha**2) for alpha in alphas])
                variable_updates = [v.assign_add(d) for v, d in zip(var_list, dirs)]
                global_step.assign_add(1)
                with tf.name_scope("summaries"):
                    with tf.name_scope("per_iteration"):
                        alpha_sum=tf.summary.scalar("alpha", alpha_norm, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
                        for (i,snr) in enumerate(abs_snrs):
                            snr_sum = tf.summary.histogram("snr_hist/"+str(i), snr, collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
        return tf.group(*variable_updates)
