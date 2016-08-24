#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QPainter>
#include <QWidget>

QT_BEGIN_NAMESPACE
class QComboBox;
class QLabel;
class QToolButton;
class QPushButton;
class QTextEdit;
QT_END_NAMESPACE
class MainWindow : public QWidget
{
    Q_OBJECT

private slots:
    void chooseSource();
    void recalculateResult();

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    void chooseImage(const QString &title, QImage *image, QToolButton *button);
    void loadImage(const QString &fileName, QImage *image, QToolButton *button);
    QPoint imagePos(const QImage &image) const;

    QToolButton *sourceButton;
    QPushButton *calculateButton;

    QImage sourceImage;
    QTextEdit *showResult;
};

#endif // MAINWINDOW_H
