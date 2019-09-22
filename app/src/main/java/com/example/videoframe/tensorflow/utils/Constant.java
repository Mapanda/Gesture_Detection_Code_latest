/*
 * Copyright (C) 2018 SoftBank Robotics Europe
 * See COPYING for the license
 */
package uni.gesturedetectiononpepper.utils;

/**
 * The constant class for TensorFlow
 */
public class Constant {

    private Constant() {
        throw new UnsupportedOperationException("Utility class");
    }

    public static final int INPUT_SIZE = 224;
    public static final int IMAGE_MEAN = 255;
    public static final float IMAGE_STD = 1;
    public static final String INPUT_NAME = "input_2";
    public static final String OUTPUT_NAME = "dense_2/Softmax";
    public static final String MODEL_FILE = "file:///android_asset/tf_model.pb";
    public static final String LABEL_FILE = "file:///android_asset/tensorflow_label_names.txt";
}
