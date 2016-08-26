#include <QtWidgets>
#include "mainwindow.h"
#include "breastcancer_predict.h"
#include "cvmatandqimage.h"
#include <cstdio>


static const QSize resultSize(480, 360);

MainWindow::MainWindow(QWidget *parent)
{
    sourceButton = new QToolButton;
    sourceButton->setIconSize(resultSize);
    calculateButton = new QPushButton(tr("&Recgnize"));
    calculateButton->setEnabled(false);
    showResult = new QTextEdit;
    showResult->setReadOnly(true);

    connect(sourceButton, SIGNAL(clicked()), this, SLOT(chooseSource()));
    connect(calculateButton, SIGNAL(clicked()), this, SLOT(recalculateResult()));
    QGridLayout *mainLayout = new QGridLayout;
    mainLayout->addWidget(sourceButton, 0, 0, 3, 1);
    mainLayout->addWidget(calculateButton, 1, 1);
    mainLayout->addWidget(showResult, 0, 2, 3, 1);
    mainLayout->setSizeConstraint(QLayout::SetFixedSize);
    setLayout(mainLayout);
    setWindowTitle(tr("breastcancer predict"));
}

MainWindow::~MainWindow()
{

}

void MainWindow::chooseSource()
{
    chooseImage(tr("Choose Source Image"), &sourceImage, sourceButton);
}

void MainWindow::chooseImage(const QString &title, QImage *image,
                                QToolButton *button)
{
    QString fileName = QFileDialog::getOpenFileName(this, title);
    if (!fileName.isEmpty())
        loadImage(fileName, image, button);
}

void MainWindow::loadImage(const QString &fileName, QImage *image,
                              QToolButton *button)
{
    sourceImg=cv::imread(fileName.toStdString());
    //image->load(fileName);
    QImage temp=QtOcv::mat2Image(sourceImg);
    cv::imshow("source img",sourceImg);
    cv::waitKey(0);
    image=&temp;

    // Scale the image to given size
    *image = image->scaled(resultSize, Qt::KeepAspectRatio);

    QImage fixedImage(resultSize, QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&fixedImage);
    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.fillRect(fixedImage.rect(), Qt::transparent);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    painter.drawImage(imagePos(*image), *image);
    painter.end();
    button->setIcon(QPixmap::fromImage(fixedImage));

    *image = fixedImage;
    calculateButton->setEnabled(true);

    recalculateResult();
}

QPoint MainWindow::imagePos(const QImage &image) const
{
    return QPoint((resultSize.width() - image.width()) / 2,
                  (resultSize.height() - image.height()) / 2);
}

cv::Mat QImage2cvMat(QImage image)
{
    cv::Mat mat;
    qDebug() << image.format();
    switch(image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}


#define mp "/home/dang/ClionProjects/breast_concer_detection/resource/svm_for_all_LBP_V2.model" //svm_for_glcm_1.model"
//#define mp "../resource/svm_for_glcm_all_2_linear.model"
//#define store_param "../resource/samples_all_pos_neg.param"
#define store_param "/home/dang/ClionProjects/breast_concer_detection/resource/samples_all_pos_neg.param"

void MainWindow::recalculateResult()
{
    showResult->clear();

    showResult->setText(tr("doing...\n"));
    //cv::waitKey(100);
    showResult->setFont(QFont( "Timers" , 18 ,  QFont::Bold));
    //CancerPredictGlcm* mcp;
    //mcp=new CancerPredictGlcm(store_param);
    CancerPredict mcp;
    //cv::Mat img=QImage2cvMat(sourceImage);
    cv::Mat img=sourceImg;
    //cv::imshow("fdfdfd",img);
    //cv::waitKey(10);
    svm_model* model=svm_load_model(mp);
    showResult->setText(tr("load model success!\n"));
    double result=mcp.predictSample(img,model);
    std::cout<<"The result is:"<<result;

    char str[10]={0};
    sprintf(str,"%lf",result);
    printf("\n%s\n",str);
    QString res;
    if(result<0){
        res=QString("result:")+"恶性"+"\n";
    }else{
        res=QString("result:")+"良性"+"\n";
    }

    res+="model info:\n";
    //svm_model* model=svm_load_model(mp);
    //svm_parameter param=model->param;
    // default values
    res+="svm_type:C_SVC\n";

    res+="kernel_type = LINEAR\n";

    res+="degree = 3\n";

    res+="gamma = 1.0/1188\n";

    res+="coef0 = 0\n";

    res+="nu = 0.5\n";
    res+="cache_size = 100\n";
    res+="C = 5\n";
    res+="eps = 1e-3\n";
    res+="p = 0.1\n";
    res+="probability = 0\n";
    res+="probability = 0\n";
    res+="nr_weight = 0\n";

    showResult->setText(res);
    //showResult->setFont(QFont( "Timers" , 18 ,  QFont::Bold));
}


