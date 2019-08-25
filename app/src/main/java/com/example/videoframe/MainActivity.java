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
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
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
            firstImageFrame = MatOnCameraframe(inputFrame);
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
            nextImageFrame = MatOnCameraframe(inputFrame);
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

    private Mat MatOnCameraframe(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        imageMat = inputFrame.rgba();
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
            Imgproc.GaussianBlur(mCol, mRgb, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT);
            //Imgproc.cvtColor(mCol, mRgb, Imgproc.COLOR_GRAY2RGBA);
            sub.apply(mRgb, mFGMask); //apply() exports a gray image by definition
            Imgproc.cvtColor(mFGMask, mCol, Imgproc.COLOR_GRAY2RGBA);
            final Size ksize = new Size(3, 3);
            Imgproc.GaussianBlur(mFGMask, imgThreshold,ksize,0);
            double thresholdedValue = Imgproc.threshold(mFGMask, imgThreshold, threshold, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU, 11);
            Log.i("*********11", String.valueOf(imgThreshold.height()));
            Log.i("*********22", String.valueOf(imgThreshold.width()));
            putText(imageMat);
            //crop the image and then proceed:
// Step HSV_Threshold0:
            Mat hsvThresholdInput = imgThreshold;


            /*boolean findContoursExternalOnly = true;
            double[] hsvThresholdHue = { 0.0, 110.88737201365191 };
            double[] hsvThresholdSaturation = { 0.0, 94 };
            double[] hsvThresholdValue = { 252.24820143884895, 255.0 };
            hsvThreshold(hsvThresholdInput, hsvThresholdHue, hsvThresholdSaturation, hsvThresholdValue, hsvThresholdOutput);
            Mat findContoursInput = hsvThresholdOutput;*/

            Mat contoursFrame = imgThreshold.clone();
            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(contoursFrame, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            int  contoursCounter = contours.size();
            Log.i("*********00:", String.valueOf(contoursCounter));
            Imgproc.cvtColor(contoursFrame, contoursFrame, Imgproc.COLOR_GRAY2BGR);

           // Mat imgContour=findContours(contoursFrame, true, findContoursOutput);
            Imgproc.drawContours(contoursFrame, contours, -1, new Scalar(255,0,0,255));
            Bitmap bitmap = matToBitmapConversion(contoursFrame);
            Bitmap cutBitmap = cutRightTop(bitmap);
            String name = "bg.jpg";
            try {
                //crop the image and save it later:
                File newfile = savebitmap(cutBitmap, name);

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
}
