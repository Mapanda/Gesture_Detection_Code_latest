package uni.gesturedetectiononpepper;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.hardware.Camera;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.aldebaran.qi.Future;
import com.aldebaran.qi.sdk.QiContext;
import com.aldebaran.qi.sdk.QiSDK;
import com.aldebaran.qi.sdk.RobotLifecycleCallbacks;
import com.aldebaran.qi.sdk.builder.SayBuilder;
import com.aldebaran.qi.sdk.builder.TakePictureBuilder;
import com.aldebaran.qi.sdk.core.QiThreadPool;
import com.aldebaran.qi.sdk.design.activity.RobotActivity;
import com.aldebaran.qi.sdk.object.camera.TakePicture;
import com.aldebaran.qi.sdk.object.conversation.Say;
import com.aldebaran.qi.sdk.object.image.EncodedImage;
import com.aldebaran.qi.sdk.object.image.EncodedImageHandle;
import com.aldebaran.qi.sdk.object.image.TimestampedImageHandle;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;

import java.io.FileOutputStream;
import java.io.IOException;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.opencv.android.Utils;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.Video;

import uni.gesturedetectiononpepper.classifier.Classifier;
import uni.gesturedetectiononpepper.classifier.ImageClassifier;
import uni.gesturedetectiononpepper.utils.Constant;
import uni.gesturedetectiononpepper.utils.UtilsClassify;
import uni.gesturedetectiononpepper.R;


public class MainActivity extends RobotActivity implements RobotLifecycleCallbacks, CameraBridgeViewBase.CvCameraViewListener2 {
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    static {
        OpenCVLoader.initDebug();
    }
   private String TAG = "Image Capture";
   private MediaPlayer playerStart;
   private MediaPlayer playerFlash;
   private  MediaPlayer playerSuccess;
    private QiContext qiContext=null;
    /**Started coding from here newly**/
    private Button backgroundExtractionButton;
    private Button gestureExtractionButton;
    Mat rgbaBilateralFrame;
    int smoothingFactor =5;
    double sigmaColor = 50.0 ;
    double sigmaSpace=100.0;
    int history =0;
    double bgThreshold=50.0;
    boolean isShadowDetected=false;
    BackgroundSubtractorKNN bgModel = Video.createBackgroundSubtractorKNN(history,bgThreshold,isShadowDetected);
    boolean isBackGroundCaptured = false;
    Mat backgroundSubtractionFrame=new Mat();
    boolean isGestureButtonClicked = false;
    private TextView gestureTextView;
    Mat rgba = new Mat();
    Classifier.Recognition bestRecognition;
    private TextView textObject;
    private ImageView gestureDetectedView;
    private Toast mToast;
    /**ended here**/
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        public void onManagerConnected(int status) {
            System.out.println("Welcome to OpenCV " + Core.VERSION);
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        QiSDK.register(this, this);
        OpenCVLoader.initDebug();
        setContentView(R.layout.activity_main);
        // verifyStoragePermissions(this);
        playerStart = MediaPlayer.create(this, R.raw.ready_sound);
        playerFlash = MediaPlayer.create(this, R.raw.automatic_camera);
        playerSuccess = MediaPlayer.create(this, R.raw.success_sound);
        // textObject = (TextView)findViewById(R.id.txt_object);
        // textObject.setVisibility(View.GONE);*/
        playerStart.start();
        gestureDetectedView = (ImageView)findViewById(R.id.imageView);
        gestureDetectedView.setVisibility(ImageView.VISIBLE);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        backgroundExtractionButton = findViewById(R.id.Background);
        backgroundExtractionButton.setEnabled(true);
        backgroundExtractionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                playerFlash.start();
                Log.i("Masking : ", "Background capture button Click Started.. ");
                isBackGroundCaptured = true;
                //getBackgroundCaptured(rgbaBilateralFrame, isBackGroundCaptured);
                getSavedImage(backgroundSubtractionFrame, "backgroundCaptured.jpg");
                Log.i("Masking : ", "Background capture button Click Finished.. ");
                backgroundExtractionButton.setEnabled(false);
                gestureExtractionButton.setEnabled(true);

            }

        });
        //for the gesture extraction
        gestureExtractionButton = findViewById(R.id.ROI);
        gestureExtractionButton.setEnabled(false);
        gestureExtractionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                playerStart.start();
                isGestureButtonClicked = true;
                // extractGesture(rgbaBilateralFrame,isGestureButtonClicked);
                Log.i("Masking : ", "Gesture capture button Click Started.. ");
                //  Imgproc.cvtColor(rgbaBilateralFrame,rgbaBilateralFrame,Imgproc.COLOR_BGR2GRAY);
                backgroundExtractionButton.setEnabled(true);
                gestureExtractionButton.setEnabled(false);
                //  findContourForBg1(rgbaBilateralFrame);
                getSavedImage(rgbaBilateralFrame, "gesturebutton.jpg");
                Bitmap maskedBitmap = matToBitmapConversion(rgbaBilateralFrame);
                String imagePredicted = classifyImage(maskedBitmap);
                Log.i("Best recognition", imagePredicted);
                putText(rgba, imagePredicted);
                Log.i("Masking : ", "Gesture capture button Click Finished.. ");
             //   playerSuccess.start();

            }

        });
    }

    public void onRobotFocusGained(QiContext qiContext) {
        this.qiContext = qiContext;
        Say say = SayBuilder.with(qiContext) // Create the builder with the context.
                .withText("All right!! Now click on '\'Background Capture' button to Start. ") // Set the text to say.
                .build(); // Build the say action.
        say.run();

    }

    public void onRobotFocusLost() {
        this.qiContext = null;
        Log.i("Capture", "Robot focus is lost.." );
    }


    public void onRobotFocusRefused(String reason) {
        Log.i(TAG,reason);
    }

    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onCameraViewStarted(int width, int height) {

    }


    public void onCameraViewStopped() {

    }
    /**
     * This method is invoked when delivery of the frame needs to be done.
     * The returned values - is a modified frame which needs to be displayed on the screen.
     * TODO: pass the parameters specifying the format of the frame (BPP, YUV or RGB and etc)
     *
     * @param inputFrame
     */

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgbaFrame = inputFrame.rgba();

     /*   double thresholdValue=50;
        Mat clippedMat = new Mat();*/
        Imgproc.cvtColor(rgbaFrame,rgbaFrame,Imgproc.COLOR_BGRA2RGB);
        rgbaBilateralFrame = rgbaFrame.clone();
        //bilateral filtering : smoothing
        Imgproc.bilateralFilter(rgbaFrame,rgbaBilateralFrame,smoothingFactor,sigmaColor,sigmaSpace);
        //cration of rectangle on the current frame:
        rgba= createRectangleOnFrame(rgbaBilateralFrame);
        //Creation of background model:
        getBackgroundCaptured(rgbaFrame,isBackGroundCaptured);
        if(isGestureButtonClicked) {
            extractGesture(backgroundSubtractionFrame,isGestureButtonClicked);
            // String classifiedResult =
            // gestureTextView.setText(classifiedResult);
        }
        //thread was added initially here. but not sure about this place.. not tested here
        try {
            Thread.sleep(100);
        } catch (InterruptedException ex) {
        }
        return rgba;
    }
    private boolean getBackgroundCaptured(Mat inputFrame,boolean isBackGroundCaptured) {
        if(!isBackGroundCaptured) {
          //  playerStart.start();
            backgroundSubtractionFrame = clipImageOnROI(inputFrame);
            isBackGroundCaptured = true;
        }
        return isBackGroundCaptured;
    }

    private void extractGesture(Mat backgroundSubtractionFrame,boolean isGestureButtonClicked) {
        //remove the background from the filtered frame:
        rgbaBilateralFrame = clipImageOnROI(rgbaBilateralFrame);
        if(!rgbaBilateralFrame.empty()) {
           removeBackgroundFromFrame(backgroundSubtractionFrame, rgbaBilateralFrame);
        }else{
            Log.i("Remove Background" ,"Frame is empty to predict the gesture. Try again");
        }
        //no idea whether this is required
        try {
            Thread.sleep(100);
        } catch (InterruptedException ex) {
        }

    }

    private void getSavedImage(Mat clippedMat,String name){
        Bitmap bitmap =matToBitmapConversion(clippedMat);
        try {

            savebitmap(bitmap,name);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private Bitmap matToBitmapConversion(Mat imageMat) {
        Bitmap bitmap = Bitmap.createBitmap(imageMat.cols(), imageMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageMat, bitmap);
        return bitmap;
    }
    //This creates a rectangle on the camera frame for the hand to get cropped.
    private Mat createRectangleOnFrame(Mat rgbaBilateralFrame){
        int w = rgbaBilateralFrame.width();
        int h = rgbaBilateralFrame.height();
        int w_rect = w*3/4;
        int h_rect = h*3/4;
        int new_width = (w+w_rect)/3;
        int new_height = (h+h_rect)/3;
        Imgproc.rectangle(rgbaBilateralFrame,  new Point(0, 0), new Point( new_width, new_height),new Scalar( 0, 255, 0 ), 5);
        return rgbaBilateralFrame;
    }
    private Mat removeBackgroundFromFrame(Mat backgroundSubtractionFrame,Mat rgbaBilateralFrame){
        Mat mRgb = new Mat();
        Mat fgMask = new Mat();
        Imgproc.GaussianBlur(rgbaBilateralFrame, backgroundSubtractionFrame, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT);
        Imgproc.cvtColor(rgbaBilateralFrame,mRgb,Imgproc.COLOR_RGBA2RGB);
        bgModel.apply(mRgb,fgMask);
        Imgproc.cvtColor(fgMask, rgbaBilateralFrame, Imgproc.COLOR_GRAY2RGBA);
        BackgroundErosion(fgMask);
        return rgbaBilateralFrame;

    }

    private void BackgroundErosion(Mat fgMask) {
        int erosion_size=5;
        final Point anchor = new Point(-1,-1);
        final int iteration=2;
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(2*erosion_size +1,2*erosion_size+1));
        //if doesn't work then change this to bilateral filtermat
        Imgproc.erode(fgMask,fgMask,kernel,anchor,iteration);
        Imgproc.dilate(fgMask,fgMask,kernel,anchor,iteration);
        final Size ksize = new Size(3,3);
        Mat skin = new Mat(fgMask.rows(),fgMask.cols(), CvType.CV_8U,new Scalar(3));
        Imgproc.GaussianBlur(fgMask,fgMask,ksize,0);
        Core.bitwise_and(fgMask,fgMask,skin);

    }

    private Mat clipImageOnROI(Mat rgbaBilateralFrame){
        Bitmap nextBitmap = matToBitmapConversion(rgbaBilateralFrame);
        Bitmap cutBitmap = clipImageMat(nextBitmap);
        rgbaBilateralFrame = bitmapToMatConversion(cutBitmap);
        return rgbaBilateralFrame;
    }

    private Mat bitmapToMatConversion(Bitmap imageBitmap){
        Mat convertedMat = new Mat();
        Bitmap bmp32 = imageBitmap.copy(Bitmap.Config.ARGB_8888,true);
        Utils.bitmapToMat(bmp32,convertedMat);
        return convertedMat;
    }

    private Bitmap clipImageMat(Bitmap origialBitmap) {
        int height = origialBitmap.getHeight();
        int width = origialBitmap.getWidth();
        int w_rect = width*3/4;
        int h_rect = height*3/4;
        int new_width = (width+w_rect)/3;
        int new_height = (height+h_rect)/3;
        //TODO; get the BOX height and width here and send it in the srcRect and dest Rect
        Bitmap cutBitmap = Bitmap.createBitmap(new_width,
                new_height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(cutBitmap);
        android.graphics.Rect desRect = new android.graphics.Rect(0, 0, new_width, new_height);
        android.graphics.Rect srcRect = new Rect(0, 0, new_width,new_height);
        canvas.drawBitmap(origialBitmap, srcRect, desRect, null);
        return cutBitmap;
    }
    public void onDestroy() {
        super.onDestroy();
        QiSDK.unregister(this, this);
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private void putText(Mat imgThreshold,String recognitionValue){
        //TODO: Add the text of detection on robot in a textfield possibly
        //Imgproc.putText(imgThreshold, recognitionValue, new Point(imgThreshold.width() / 2,30), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(255,255,255),3);
    }

    private  void savebitmap(Bitmap bmp , String name) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 60, bytes);
        verifyStoragePermissions(this);
        File f = new File(Environment.getExternalStorageDirectory()
                + File.separator + name);
        f.createNewFile();
        FileOutputStream fo = new FileOutputStream(f);
        fo.write(bytes.toByteArray());
        fo.close();

    }
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
    public String classifyImage(Bitmap bitmap) {
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
        Mat resizedMat = bitmapToMatConversion(resizedBitmap);
        runOnUiThread(() -> gestureDetectedView.setImageBitmap(bitmap));
        getSavedImage(resizedMat,"Classify.jpg");
        List<Classifier.Recognition> results = classifier.recognizeImage(resizedBitmap);
        bestRecognition = new Classifier.Recognition("None", "NoGesture", 0.0f);
        int count =0;
        for (Classifier.Recognition recognition :
                results) {
            count++;
            if (recognition.getConfidence() > bestRecognition.getConfidence())
                bestRecognition = recognition;

        }
        displayGestureOnScreen();

        // layoutResult(UtilsClassify.getResizedBitmap(bitmap, 1400, 900, true));
     //   Mat textMap = bitmapToMatConversion(resizedBitmap);

/*        Log.i("Prediction : ****", String.valueOf(bestRecognition.getConfidence()));
        putText(textMap,String.valueOf(bestRecognition.getConfidence()));
        putText(textMap,bestRecognition.getTitle());
        Log.i("Count : ",String.valueOf (count));*/
        Log.i("Name : " , bestRecognition.getTitle());
        // String classifiedResult = processResult(bestRecognition);
        //String gestureDetected = "Gesture detected: "+" " + bestRecognition.getTitle() + classifiedResult;
        float confidence = bestRecognition.getConfidence() * 100;
        playerSuccess.start();
        new AlertDialog.Builder(this)
                .setTitle("Information")
                .setMessage("Detected Gesture is :" + bestRecognition.getTitle()+"("+confidence+")%")
                .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        // Continue with delete operation
                    }
                }).show();
      //  actionOnDetectedGesture(bestRecognition);

     /*   Say say = SayBuilder.with(qiContext) // Create the builder with the context.
                .withText("Gesture detected as.. " + name) // Set the text to say.
                .build(); // Build the say action.

        // Execute the action.
        say.run();*/

        //  QiThreadPool.run(() -> processResult(bestRecognition));

        return bestRecognition.getTitle();
    }

    private void displayGestureOnScreen() {
        float confidence = bestRecognition.getConfidence() * 100;
        String name = bestRecognition.getTitle();
        String message="";
        if(confidence < 99.0) {
            message = "Gesture is not properly detected.Try again!!";
       }
        else{
              if (mToast != null && mToast.getView().isShown())
                     mToast.cancel(); // Close the toast if it is already open
                     message = "Detected Gesture is :" + bestRecognition.getTitle() + "(" + confidence + ")";

       }
        int duration = Toast.LENGTH_LONG;
        mToast = Toast.makeText(this, message, duration);
        mToast.show();
    }

    private void processResult(Classifier.Recognition recognition) {
        String name = recognition.getTitle();
        float confidence = recognition.getConfidence() * 100;
        textObject.setText(getString(R.string.txt_object_text,name, String.format(Locale.getDefault(), "%.2f",confidence)));
        textObject.setVisibility(View.VISIBLE);
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle(R.string.title_message)
                .setMessage(R.string.title_detected_gesture)
                .setPositiveButton(R.string.txt_object_text + name, (dialog, which) -> {
                    startActivityForResult(new Intent(Settings.ACTION_LOCALE_SETTINGS), 0);
                })
                .setIcon(android.R.drawable.btn_dialog)
                .show();
         Say say = SayBuilder.with(this.qiContext) // Create the builder with the context.
                .withText("Gesture detected as.. " + name) // Set the text to say.
                .build(); // Build the say action.

        // Execute the action.
        say.run();
       // playerSuccess.start();


    }

    private void actionOnDetectedGesture(Classifier.Recognition recognition){
        String name = recognition.getTitle();
        Float confidence = recognition.getConfidence();
        if(name.equalsIgnoreCase("L")){


        }else if(name.equalsIgnoreCase("Peace")){

            String locationName = " university of Rostock";
            String locationDescription = "Department of Computer Science. Building AE22";


            Say sayName = SayBuilder.with(qiContext).withText("This Location is "+locationName).build();
            Say sayDescription = SayBuilder.with(qiContext)
                    .withText(locationDescription)
                    .build();

            sayName.run();
            sayDescription.run();

        }else if(name.equalsIgnoreCase("Palm")){

        }else if(name.equalsIgnoreCase("Fist")){

        }else if(name.equalsIgnoreCase("Okay")){

        }else{
              /*  if(confidence < 99.0){
                    displayGestureOnScreen();
                }*/
        }


    }

/*    private void holdAbilities(QiContext qiContext) {
        // Build the holder for the abilities.
        Holder holder = HolderBuilder.with(qiContext)
                .withAutonomousAbilities(
                        AutonomousAbilitiesType.BACKGROUND_MOVEMENT,
                        AutonomousAbilitiesType.BASIC_AWARENESS,
                        AutonomousAbilitiesType.AUTONOMOUS_BLINKING
                )
                .build();

        // Hold the abilities asynchronously.
        Future<Void> holdFuture = holder.async().hold();
    }*/
}