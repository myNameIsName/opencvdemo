package com.opencvdemo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class ContrastActivity extends AppCompatActivity {

    private ImageView mIvGrayFace1;
    private ImageView mIvGrayFace2;
    private Mat mGrayFace1, mGrayFace2;

    private CascadeClassifier classifier;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_contrast);
        mIvGrayFace1 = findViewById(R.id.iv_face_1);
        mIvGrayFace2 = findViewById(R.id.iv_face_2);
        findViewById(R.id.btn_gray_face).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //检测人脸是耗时的操作
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        getImage1Face();
                        getImage2Face();
                    }
                }).start();
            }
        });
        findViewById(R.id.btn_contras_face).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mGrayFace1 == null || mGrayFace2 == null) {
                    Toast.makeText(getApplicationContext(), "请先获取灰度图", Toast.LENGTH_SHORT).show();
                } else {
                    mGrayFace1.convertTo(mGrayFace1, CvType.CV_32F);
                    Mat mat2 = new Mat();
                    Imgproc.resize(mGrayFace2, mat2, new Size(mGrayFace1.cols(), mGrayFace1.rows()));
                    mat2.convertTo(mat2, CvType.CV_32F);
                    double target = Imgproc.compareHist(mGrayFace1, mat2, Imgproc.CV_COMP_CORREL);
                    Toast.makeText(getApplicationContext(), "相似度：" + target, Toast.LENGTH_SHORT).show();
                }
            }
        });
        initClassifier();
    }

    private void initClassifier() {
        try {
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void getImage1Face() {
        Bitmap mBitmap1 = BitmapFactory.decodeResource(getResources(), R.drawable.img1);
        Mat mat1 = new Mat();
        Mat mat11 = new Mat();
        Utils.bitmapToMat(mBitmap1, mat1);
        Imgproc.cvtColor(mat1, mat11, Imgproc.COLOR_BGR2GRAY);
        Rect[] object = detectObjectImage(mat11);//图片上只有一个人，当然也就只能识别出一张人脸，所以我们直接取第一个就行了
        if (object != null && object.length > 0) {
            Mat mat = mat11.submat(object[0]);
            mGrayFace1 = mat;
            final Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, bitmap);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mIvGrayFace1.setImageBitmap(bitmap);
                }
            });
        }
    }

    private void getImage2Face() {
        Bitmap mBitmap1 = BitmapFactory.decodeResource(getResources(), R.drawable.img2);
        Mat mat1 = new Mat();
        Utils.bitmapToMat(mBitmap1, mat1);

        Mat mat11 = new Mat();
        Imgproc.cvtColor(mat1, mat11, Imgproc.COLOR_BGR2GRAY);

        Rect[] object = detectObjectImage(mat11);
        if (object != null && object.length > 0) {
            Mat mat = mat11.submat(object[0]);
            mGrayFace2 = mat;
            final Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, bitmap);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mIvGrayFace2.setImageBitmap(bitmap);
                }
            });
        }
    }

    public Rect[] detectObjectImage(Mat gray) {
        MatOfRect faces = new MatOfRect();
        classifier.detectMultiScale(gray, faces);
        return faces.toArray();
    }

}
