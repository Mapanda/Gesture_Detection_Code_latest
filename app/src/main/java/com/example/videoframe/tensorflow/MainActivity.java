package com.example.videoframe.tensorflow;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import com.example.videoframe.R;
import com.example.videoframe.tensorflow.classifier.Classifier;
import com.example.videoframe.tensorflow.classifier.ImageClassifier;
import com.example.videoframe.tensorflow.utils.Constant;
import com.example.videoframe.tensorflow.utils.UtilsClassify;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2{
    private CameraBridgeViewBase mOpenCvCameraView;
    public  CvCameraViewListener2 camListener;
    private BackgroundSubtractorMOG2 sub = Video.createBackgroundSubtractorMOG2();
    // Storage Permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    Mat  imageMat;
    Mat firstImageFrame ;
    Mat nextImageFrame ;
    Mat imgThreshold = new Mat();
    boolean isBackGroundCaptured =false;
    private Mat mFGMask = new Mat();
    double blurValue = 41.0;  // GaussianBlur parameter
    private ArrayList<MatOfPoint> findContoursOutput = new ArrayList<>();
    private Mat hsvThresholdOutput = new Mat();
    static {
        OpenCVLoader.initDebug();
    }
    Classifier.Recognition bestRecognition;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                   // imageMat=new Mat();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("APP", "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        String s = "CAMERA";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
                }
        }
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }


    /**
     * This method is invoked when camera preview has started. After this method is invoked
     * the frames will start to be delivered to client via the onCameraFrame() callback.
     *
     * @param width  -  the width of the frames that will be delivered
     * @param height - the height of the frames that will be delivered
     */
    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    /**
     * This method is invoked when camera preview has been stopped for some reason.
     * No frames will be delivered via onCameraFrame() callback after this method is called.
     */
    @Override
    public void onCameraViewStopped() {

    }

    /**
     * This method is invoked when delivery of the frame needs to be done.
     * The returned values - is a modified frame which needs to be displayed on the screen.
     * TODO: pass the parameters specifying the format of the frame (BPP, YUV or RGB and etc)
     *
     * @param inputFrame
     */
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Highgui
    int count =0;
        if(!isBackGroundCaptured) {
            count = initialBackgroundCapture(inputFrame, count);
        }else{
            count = continuousBackgroundCapture(inputFrame);
        }
        if(count==0){
        backgroundSubtraction(firstImageFrame,nextImageFrame);}
        return imageMat;
    }

    private int continuousBackgroundCapture(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        int count;
        count=0;
        nextImageFrame = matOnCameraframe(inputFrame);
        Bitmap nextBitmap = matToBitmapConversion(nextImageFrame);
        Bitmap cutBitmap = cutRightTop(nextBitmap);
        String name = "requiredImage.jpg";
        try {
            File newfile = savebitmap(cutBitmap, name);

        } catch (IOException e) {
            e.printStackTrace();
        }
        return count;
    }

    private int initialBackgroundCapture(CameraBridgeViewBase.CvCameraViewFrame inputFrame, int count) {
        firstImageFrame = matOnCameraframe(inputFrame);
        Bitmap bitmapFirst = matToBitmapConversion(firstImageFrame);
        Bitmap cutBitmap = cutRightTop(bitmapFirst);
        String name = "removeBackround.jpg";
        try {
            File newfile = savebitmap(cutBitmap, name);
            Log.i("*********22222", newfile.getAbsolutePath());
            isBackGroundCaptured = true;
            count++;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return count;
    }

    private Bitmap matToBitmapConversion(Mat imageMat) {
        Bitmap bitmap = Bitmap.createBitmap(imageMat.cols(), imageMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageMat, bitmap);
        return bitmap;
    }
    private Mat bitmapToMatConversion(Bitmap imageBitmap){
        Mat convertedMat = new Mat();
        Bitmap bmp32 = imageBitmap.copy(Bitmap.Config.ARGB_8888,true);
        Utils.bitmapToMat(bmp32,convertedMat);
        return convertedMat;
    }
    private Mat matOnCameraframe(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat imageMat1 = inputFrame.rgba();
        Imgproc.cvtColor(imageMat1,imageMat1,Imgproc.COLOR_BGRA2RGB);
        imageMat = imageMat1.clone();
        Imgproc.bilateralFilter(imageMat1,imageMat,10,250,50); //smoothing filter
        Imgproc.cvtColor(imageMat1,imageMat1,Imgproc.COLOR_RGB2RGBA);
        int w = imageMat.width();
        int h = imageMat.height();
        int w_rect = w*3/4;
        int h_rect = h*3/4;
        int new_width = (w+w_rect)/3;
        int new_height = (h+h_rect)/3;
        Imgproc.rectangle(imageMat,  new Point(new_width, new_height), new Point( 0, 0),new Scalar( 255, 0, 0 ), 5);
        return imageMat;
    }
    private Bitmap cutRightTop( Bitmap origialBitmap) {
        int height = origialBitmap.getHeight();
        int width = origialBitmap.getWidth();
        int w_rect = width*3/4;
        int h_rect = height*3/4;
        int new_width = (width+w_rect)/3;
        int new_height = (height+h_rect)/3;
        //TODO; get the BOX height and width here and send it in the srcRect and dest Rect
        Bitmap cutBitmap = Bitmap.createBitmap(origialBitmap.getWidth(),
                origialBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(cutBitmap);
        Rect desRect = new Rect(0, 0, new_width, new_height);
        Rect srcRect = new Rect(0, 0, new_width,new_height);
        canvas.drawBitmap(origialBitmap, srcRect, desRect, null);
        return cutBitmap;
    }
    public void backgroundSubtraction(Mat mRgb,Mat mCol) {
            double threshold =16.0;
            int erosion_size = 5;
            Imgproc.GaussianBlur(mCol, mRgb, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT);
            Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(2*erosion_size +1,2*erosion_size+1));
            sub.apply(mRgb, mFGMask); //apply() exports a gray image by definition
            Imgproc.cvtColor(mFGMask, mCol, Imgproc.COLOR_GRAY2RGBA);
            final Size ksize = new Size(3, 3);
            Imgproc.erode(mFGMask,imgThreshold,element);
            Imgproc.GaussianBlur(mFGMask, imgThreshold,ksize,0);
            double thresholdedValue = Imgproc.threshold(mFGMask, imgThreshold, threshold, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU, 11);
            findContourForBg();
    }

    private void findContourForBg() {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        int  contoursCounter = contours.size();
        Bitmap bitmap = matToBitmapConversion(imgThreshold);
        Bitmap cutBitmap = cutRightTop(bitmap);
        Mat croppedMat = bitmapToMatConversion(cutBitmap);
        Bitmap cutBitmapped = matToBitmapConversion(croppedMat);
        classifyImage(cutBitmapped);
        String name = "bg.jpg";
        try {
            //crop the image and save it later:
            File newfile = savebitmap(cutBitmapped, name);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Mat contoursFrame = croppedMat.clone();
        Imgproc.cvtColor(croppedMat, contoursFrame, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.findContours(contoursFrame, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.cvtColor(contoursFrame, contoursFrame, Imgproc.COLOR_GRAY2BGRA);


          if (contours.size() > 0)  // Minimum size allowed for consideration
          {
              for (int contourIdx=0; contourIdx < contours.size(); contourIdx++ ) {
                  MatOfPoint  temp = contours.get(contourIdx);
                  Imgproc.drawContours(contoursFrame, contours, -1, new Scalar(0, 255, 0),5);
          }
        }
        Bitmap cutBitmapped1= matToBitmapConversion(contoursFrame);
        String name1 = "bg1.jpg";
        try {
            //crop the image and save it later:
           File newfile = savebitmap(cutBitmapped1, name1);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void putText(Mat imgThreshold,String recognitionValue){
        //TODO: Add the text of detection on robot in a textfield possibly
        Imgproc.putText(imgThreshold, recognitionValue, new Point(imgThreshold.width() / 2,30), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(0,255,255),3);
    }

    private  File savebitmap(Bitmap bmp , String name) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 60, bytes);
        verifyStoragePermissions(this);
        File f = new File(Environment.getExternalStorageDirectory()
                + File.separator + name);
        f.createNewFile();
        FileOutputStream fo = new FileOutputStream(f);
        fo.write(bytes.toByteArray());
        fo.close();
        return f;
    }
    /**
     * Checks if the app has permission to write to device storage
     *
     * If the app does not has permission then the user will be prompted to grant permissions
     *
     * @param activity
     */
    public static void verifyStoragePermissions(Activity activity) {
        // Check if we have write permission
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    public void classifyImage(Bitmap bitmap) {
        Classifier classifier =
                ImageClassifier.create(
                        getAssets(),
                        Constant.MODEL_FILE,
                        Constant.LABEL_FILE,
                        Constant.INPUT_SIZE,
                        Constant.IMAGE_MEAN,
                        Constant.IMAGE_STD,
                        Constant.INPUT_NAME,
                        Constant.OUTPUT_NAME);

        Bitmap resizedBitmap = UtilsClassify.getResizedBitmap(bitmap, Constant.INPUT_SIZE, Constant.INPUT_SIZE, false);
        List<Classifier.Recognition> results = classifier.recognizeImage(resizedBitmap);
        bestRecognition = new Classifier.Recognition("test", "testObject", 0.0f);
        int count =0;
        for (Classifier.Recognition recognition :
                results) {
           count++;
            if (recognition.getConfidence() > bestRecognition.getConfidence())
                bestRecognition = recognition;
            Mat textMap = bitmapToMatConversion(resizedBitmap);

            Log.i("Prediction : ****", String.valueOf(bestRecognition.getConfidence()));
            putText(textMap,String.valueOf(bestRecognition.getConfidence()));
            putText(textMap,bestRecognition.getTitle());
            Log.i("Count : ",String.valueOf (count));
            Log.i("Name : " , bestRecognition.getTitle());
        }
        processResult(bestRecognition);
    }

    private void processResult(Classifier.Recognition recognition) {
        String name = bestRecognition.getTitle();
        float confidence = recognition.getConfidence() * 100;
        String bookmark;
        if (confidence > 15 && confidence < 40) {
            bookmark = "classify20";
        } else if (confidence > 40 && confidence < 60) {
            bookmark = "classify50";
        } else if (confidence > 60 && confidence < 80) {
            bookmark = "classify70";
        } else if (confidence > 80) {
            bookmark = "classify90";
        } else {
            bookmark = "failClassify";
        }
    }
}
