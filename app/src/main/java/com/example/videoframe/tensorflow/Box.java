package com.example.videoframe;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

public class Box extends View {
    private Paint paint = new Paint();
    Box(Context context) {
        super(context);
    }

    @Override
    protected void onDraw(Canvas canvas) { // Override the onDraw() Method
        super.onDraw(canvas);

        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.GREEN);
        paint.setStrokeWidth(10);

        //center
        int x0 = canvas.getWidth()/2;
        int y0 = canvas.getHeight()/2;
        int dx = canvas.getHeight()/4;
        int dy = canvas.getHeight()/4;
     /*   int finalx= x0-dx;
        int finaly = y0-dy;
        int finalrightx =x0+dx;
        int finalrighty = y0+dy;
       int x1 =1000;
        int y1 = 1000;
        int width = 500;
        int height = 500;*/
        //draw guide box
        canvas.drawRect(x0, y0,0 , 0, paint);
        // canvas.drawRect(x1, y1, width, height, paint);
        System.out.println("*********************BOX STARTED**********************");
   /*     System.out.println("Size-x0:"+x1);
        System.out.println("Size-y0:"+y1);
        System.out.println("Size-dx:"+width);
        System.out.println("Size-dy:"+height);*/
        System.out.println("Size-x0:"+x0);
        System.out.println("Size-y0:"+y0);
       /* System.out.println("Size-dx:"+finalrightx);
        System.out.println("Size-dy:"+finalrighty);*/
        System.out.println("*********************BOX ENDED**********************");
    }
    //Programmatically photos for screenshots:
    //https://stackoverflow.com/questions/2661536/how-to-programmatically-take-a-screenshot-on-android
}
