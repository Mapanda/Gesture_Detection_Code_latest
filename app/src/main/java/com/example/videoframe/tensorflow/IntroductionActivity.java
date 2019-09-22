package uni.gesturedetectiononpepper;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.support.v4.app.ActivityCompat;
import android.util.Log;

import com.aldebaran.qi.sdk.QiContext;
import com.aldebaran.qi.sdk.QiSDK;
import com.aldebaran.qi.sdk.RobotLifecycleCallbacks;
import com.aldebaran.qi.sdk.builder.SayBuilder;
import com.aldebaran.qi.sdk.design.activity.RobotActivity;
import com.aldebaran.qi.sdk.object.conversation.Say;

import org.opencv.core.Mat;

import java.util.Locale;

import uni.gesturedetectiononpepper.R;

import static uni.gesturedetectiononpepper.IntroductionActivity.verifyStoragePermissions;

public class IntroductionActivity extends RobotActivity implements RobotLifecycleCallbacks {
    // The button used to start take picture action.
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    Mat imageMat;
    private QiContext qiContext;
    private static int SPLASH_SCREEN_TIME_OUT=7500;
    private String TAG="Introduction Activity";
    boolean isCheckLanguageDone = true;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        isCheckLanguageDone=checkLanguage();
        QiSDK.register(this,this);
        setContentView(R.layout.activity_introduction);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
            }
        }
        verifyStoragePermissions(this);
       if(isCheckLanguageDone){
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent i=new Intent(IntroductionActivity.this,
                        MainActivity.class);
                //Intent is used to switch from one activity to another.
                startActivity(i);
                //invoke the SecondActivity.
                finish();
                //the current activity will get finished.
            }
        }, SPLASH_SCREEN_TIME_OUT);
       }else{
           AlertDialog.Builder builder = new AlertDialog.Builder(this);
           builder.setTitle(R.string.title_dialog_access_error)
                   .setMessage(R.string.Not_allowed_to_access)
                   .setPositiveButton(R.string.button_dialog_settings, (dialog, which) -> {
                       startActivityForResult(new Intent(Settings.ACTION_LOCALE_SETTINGS), 0);
                   })
                   .setIcon(android.R.drawable.ic_dialog_alert)
                   .show();
       }
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
    /**
     * Called when focus is gained
     *
     * @param qiContext the robot context
     */
    @Override
    public void onRobotFocusGained(QiContext qiContext) {
        this.qiContext = qiContext;
        if(isCheckLanguageDone) {
            Say say = SayBuilder.with(qiContext) // Create the builder with the context.
                    .withText("Welcome, this is Pepper!! I can guide you. Let me show you how to Communicate with me.. ") // Set the text to say.
                    .build(); // Build the say action.

            // Execute the action.
            say.run();
        }

    }
    @Override
    protected void onDestroy() {
        // Unregister the RobotLifecycleCallbacks for this Activity.
        QiSDK.unregister(this, this);
        super.onDestroy();
    }
    /**
     * Called when focus is lost
     */
    @Override
    public void onRobotFocusLost() {
        Log.d(TAG, "Testing ");
    }

    /**
     * Called when focus is refused
     *
     * @param reason the reason
     */
    @Override
    public void onRobotFocusRefused(String reason) {
        Log.d(TAG, reason);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        checkLanguage();
    }

    private boolean checkLanguage(){
        if(!Locale.getDefault().getDisplayLanguage().equals("English")){
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle(R.string.title_dialog_language_check)
                    .setMessage(R.string.Message_dialog_language_check)
                    .setPositiveButton(R.string.button_dialog_settings, (dialog, which) -> {
                        startActivityForResult(new Intent(Settings.ACTION_LOCALE_SETTINGS), 0);
                    })
                    .setIcon(android.R.drawable.ic_dialog_alert)
                    .show();
            isCheckLanguageDone = false;
        } else {
            QiSDK.register(this, this);
        }
        return isCheckLanguageDone;
    }
}
