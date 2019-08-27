package com.example.videoframe;

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
import android.view.View;
import android.view.WindowManager;

import com.example.videoframe.tensorflow.classifier.Classifier;
import com.example.videoframe.tensorflow.classifier.ImageClassifier;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
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
import java.util.Locale;

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
        //mOpenCvCameraView.enableView();
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
    int count =0;
        if(!isBackGroundCaptured) {
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
        }else{
            count=0;
            nextImageFrame = matOnCameraframe(inputFrame);
            Bitmap nextBitmap = matToBitmapConversion(nextImageFrame);
            Bitmap cutBitmap = cutRightTop(nextBitmap);

            String name = "requiredImage.jpg";
            try {
                File newfile = savebitmap(cutBitmap, name);
                Log.i("*********22222", newfile.getAbsolutePath());

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if(count==0)
        backgroundSubtraction(firstImageFrame,nextImageFrame);
        return imageMat;
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
        int w_rect = w*3/4; // or 640
        int h_rect = h*3/4; // or 480
        int new_width = (w+w_rect)/3;
        int new_height = (h+h_rect)/3;
        Imgproc.rectangle(imageMat,  new Point(new_width, new_height), new Point( 0, 0),new Scalar( 255, 0, 0 ), 5);
        return imageMat;
    }
    private Bitmap cutRightTop( Bitmap origialBitmap) {
        int height = origialBitmap.getHeight();
        int width = origialBitmap.getWidth();
        int w_rect = width*3/4; // or 640
        int h_rect = height*3/4; // or 480
        int new_width = (width+w_rect)/3;
        int new_height = (height+h_rect)/3;
        //TODO; get the BOX height and width here and send it in the srcRect and dest Rect
        Bitmap cutBitmap = Bitmap.createBitmap(origialBitmap.getWidth(),
                origialBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(cutBitmap);
        Rect desRect = new Rect(0, 0, new_width, new_height);
        Rect srcRect = new Rect(0, 0, new_width,new_height);
        Log.i("*********1", String.valueOf(w_rect));
        Log.i("*********2", String.valueOf(h_rect));
        canvas.drawBitmap(origialBitmap, srcRect, desRect, null);

        return cutBitmap;
    }
    public void backgroundSubtraction(Mat mRgb,Mat mCol) {
            double threshold =16.0;
            int erosion_size = 5;
            Imgproc.GaussianBlur(mCol, mRgb, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT);
            //Imgproc.cvtColor(mCol, mRgb, Imgproc.COLOR_GRAY2RGBA);
            Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(2*erosion_size +1,2*erosion_size+1));
            sub.apply(mRgb, mFGMask); //apply() exports a gray image by definition
            Imgproc.cvtColor(mFGMask, mCol, Imgproc.COLOR_GRAY2RGBA);
            final Size ksize = new Size(3, 3);
            Imgproc.erode(mFGMask,imgThreshold,element);
           // Mat skin= skinDetection(imgThreshold);
            Imgproc.GaussianBlur(mFGMask, imgThreshold,ksize,0);
            double thresholdedValue = Imgproc.threshold(mFGMask, imgThreshold, threshold, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU, 11);
            Log.i("*********11", String.valueOf(imgThreshold.height()));
            Log.i("*********22", String.valueOf(imgThreshold.width()));
            putText(imageMat);
            //crop the image and then proceed:
            // Step HSV_Threshold0:
          //  Mat hsvThresholdInput = imgThreshold;


            /*boolean findContoursExternalOnly = true;
            double[] hsvThresholdHue = { 0.0, 110.88737201365191 };
            double[] hsvThresholdSaturation = { 0.0, 94 };
            double[] hsvThresholdValue = { 252.24820143884895, 255.0 };
            hsvThreshold(hsvThresholdInput, hsvThresholdHue, hsvThresholdSaturation, hsvThresholdValue, hsvThresholdOutput);
            Mat findContoursInput = hsvThresholdOutput;*/

           // Mat contoursFrame = imgThreshold.clone();
           findContourForBg();
    }
    public Mat skinDetection(Mat src) {
        // define the upper and lower boundaries of the HSV pixel
        // intensities to be considered 'skin'
        Scalar lower = new Scalar(0, 48, 80);
        Scalar upper = new Scalar(20, 255, 255);

        // Convert to HSV
        Mat hsvFrame = new Mat(src.rows(), src.cols(), CvType.CV_8U, new Scalar(3));
        Imgproc.cvtColor(src, hsvFrame, Imgproc.COLOR_RGB2HSV, 3);

        // Mask the image for skin colors
        Mat skinMask = new Mat(hsvFrame.rows(), hsvFrame.cols(), CvType.CV_8U, new Scalar(3));
        Core.inRange(hsvFrame, lower, upper, skinMask);
//        currentSkinMask = new Mat(hsvFrame.rows(), hsvFrame.cols(), CvType.CV_8U, new Scalar(3));
//        skinMask.copyTo(currentSkinMask);

        // apply a series of erosions and dilations to the mask
        // using an elliptical kernel
        final Size kernelSize = new Size(11, 11);
        final Point anchor = new Point(-1, -1);
        final int iterations = 2;

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, kernelSize);
        Imgproc.erode(skinMask, skinMask, kernel, anchor, iterations);
        Imgproc.dilate(skinMask, skinMask, kernel, anchor, iterations);

        // blur the mask to help remove noise, then apply the
        // mask to the frame
        final Size ksize = new Size(3, 3);

        Mat skin = new Mat(skinMask.rows(), skinMask.cols(), CvType.CV_8U, new Scalar(3));
        Imgproc.GaussianBlur(skinMask, skinMask, ksize, 0);
        Core.bitwise_and(src, src, skin, skinMask);

        return skin;
    }

    private void findContourForBg() {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        // Imgproc.findContours(contoursFrame, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        int  contoursCounter = contours.size();
        Log.i("*********00:", String.valueOf(contoursCounter));
        // Mat imgContour=findContours(contoursFrame, true, findContoursOutput);
        //Imgproc.drawContours(contoursFrame, contours, -1, new Scalar(255,0,0,255));
        Bitmap bitmap = matToBitmapConversion(imgThreshold);
        Bitmap cutBitmap = cutRightTop(bitmap);
        Mat croppedMat = bitmapToMatConversion(cutBitmap);
        Mat contoursFrame = croppedMat.clone();
        Imgproc.cvtColor(croppedMat, contoursFrame, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.findContours(contoursFrame, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.cvtColor(contoursFrame, contoursFrame, Imgproc.COLOR_GRAY2BGRA);


      /*  double maxArea = 0;
        MatOfPoint max_contour = new MatOfPoint();
        List<MatOfPoint> contour = new ArrayList<MatOfPoint>();
        List<MatOfInt> hull = new ArrayList<MatOfInt>();
        Iterator<MatOfPoint> iterator = contours.iterator();
        while (iterator.hasNext()){
             contour = iterator.next();
            double area = Imgproc.contourArea(contour);
            if(area > maxArea){
                maxArea = area;
                max_contour = contour;
            }
        }
        Size size = max_contour.size();*/

        /*size.
        MatOfPoint res=  contours.get();
        MatOfInt hull= Imgproc.convexHull(contour, hull);
        Mat contourDrawMat = new Mat();

        Imgproc.drawContours(contourDrawMat, hull, 0, new Scalar(0, 255, 0),2);
        Imgproc.drawContours(contourDrawMat, res, 0, new Scalar(0, 255, 0),2);*/


          if (contours.size() > 0)  // Minimum size allowed for consideration
          {
              for (int contourIdx=0; contourIdx < contours.size(); contourIdx++ ) {
                  MatOfPoint  temp = contours.get(contourIdx);

                Imgproc.drawContours(contoursFrame, contours, -1, new Scalar(0, 255, 0),5);
          }
        }

        Bitmap cutBitmapped = matToBitmapConversion(contoursFrame);
        String name = "bg1.jpg";
        try {
            //crop the image and save it later:
            File newfile = savebitmap(cutBitmapped, name);

        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            Thread.sleep(30);
        } catch (InterruptedException ex) {

        }
    }

    private void putText(Mat imgThreshold){
        //TODO: Add the text of detection on robot in a textfield possibly
        Imgproc.putText(imgThreshold, "Frame", new Point(imgThreshold.width() / 2,30), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(0,255,255),3);
    }
    private void hsvThreshold(Mat input, double[] hue, double[] sat, double[] val, Mat out) {
        Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HSV);
        Core.inRange(out, new Scalar(hue[0], sat[0], val[0]), new Scalar(hue[1], sat[1], val[1]), out);
    }
    /**
     * Sets the values of pixels in a binary image to their distance to the nearest black pixel.
     *  maskSize
     *            the size of the mask.
     * output
     * @param input
*            The image on which to perform the Distance Transform.
     */
    private Mat findContours(Mat input, boolean externalOnly, List<MatOfPoint> contours) {
        MatOfPoint2f approxCurve = new MatOfPoint2f();

        //For each contour found
        for (int i=0; i<contours.size(); i++)
        {
            //Convert contours(i) from MatOfPoint to MatOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

            // Get bounding rect of contour
            org.opencv.core.Rect rect = Imgproc.boundingRect(points);

            // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
            Imgproc.rectangle(input, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(255, 0, 0, 255), 3);

        }
        return input;
    }

    public  File savebitmap(Bitmap bmp , String name) throws IOException {
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
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);

        Bitmap resizedBitmap = Utils.getResizedBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        List<Classifier.Recognition> results = classifier.recognizeImage(resizedBitmap);
        bestRecognition = new Classifier.Recognition("test", "testObject", 0.0f);

        for (Classifier.Recognition recognition :
                results) {
            if (recognition.getConfidence() > bestRecognition.getConfidence())
                bestRecognition = recognition;
        }

        layoutResult(Utils.getResizedBitmap(bitmap, 1400, 900, true));
        QiThreadPool.run(() -> processResult(bestRecognition));
    }

    private void processResult(Classifier.Recognition recognition) {
        String name = bestRecognition.getTitle();
        float confidence = recognition.getConfidence() * 100;
        String bookmark;

        txtObject.setText(getString(R.string.txt_object_text, name, String.format(Locale.getDefault(), "%.2f", confidence)));
        txtObject.setVisibility(View.VISIBLE);

        playerSuccess.start();

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

        isScanning.set(false);

        if (pepperHolder != null)
            pepperHolder.release();

        RobotUtils.goToBookmark(qiChatbot, bookmarks, bookmark, name).getValue();
    }
}
