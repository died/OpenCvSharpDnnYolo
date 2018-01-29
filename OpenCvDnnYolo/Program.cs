using System;
using System.Diagnostics;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OpenCvDnnYolo
{
    class Program
    {
        private static readonly string[] Labels = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
        private static readonly Scalar[] Colors = Enumerable.Repeat(false, 20).Select(x => Scalar.RandomColor()).ToArray();
        static void Main()
        {
            var file = "bali.jpg";
            // https://pjreddie.com/darknet/yolo/
            var cfg = "yolo-voc.cfg";
            var model = "yolo-voc.weights"; //YOLOv2 544x544
            var threshold = 0.3;

            var org = Cv2.ImRead(file);
            var w = org.Width;
            var h = org.Height;
            //setting blob, parameter are important
            var blob = CvDnn.BlobFromImage(org, 1 / 255.0, new Size(544, 544), new Scalar(), true, false);
            var net = CvDnn.ReadNetFromDarknet(cfg, model);
            net.SetInput(blob, "data");

            Stopwatch sw = new Stopwatch();
            sw.Start();
            //forward model
            var prob = net.Forward();
            sw.Stop();
            Console.WriteLine($"Runtime:{sw.ElapsedMilliseconds} ms");

            /* YOLO2 VOC output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~24 : class probability */
            const int prefix = 5;   //skip 0~4

            for (int i = 0; i < prob.Rows; i++)
            {
                var confidence = prob.At<float>(i, 4);
                if (confidence > threshold)
                {
                    //get classes probability
                    Cv2.MinMaxLoc(prob.Row[i].ColRange(prefix, prob.Cols), out _, out Point max);
                    var classes = max.X;
                    var probability = prob.At<float>(i, classes + prefix);

                    if (probability > threshold) //more accuracy
                    {
                        //get center and width/height
                        var centerX = prob.At<float>(i, 0) * w;
                        var centerY = prob.At<float>(i, 1) * h;
                        var width = prob.At<float>(i, 2) * w;
                        var height = prob.At<float>(i, 3) * h;
                        //label formating
                        var label = $"{Labels[classes]} {probability * 100:0.00}%";
                        Console.WriteLine($"confidence {confidence * 100:0.00}% {label}");
                        var x1 = (centerX - width / 2) < 0 ? 0 : centerX - width / 2; //avoid left side over edge
                        //draw result
                        org.Rectangle(new Point(x1, centerY - height / 2), new Point(centerX + width / 2, centerY + height / 2), Colors[classes], 2);
                        var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);
                        Cv2.Rectangle(org, new Rect(new Point(x1, centerY - height / 2 - textSize.Height - baseline),
                                new Size(textSize.Width, textSize.Height + baseline)), Colors[classes], Cv2.FILLED);
                        Cv2.PutText(org, label, new Point(x1, centerY - height / 2-baseline), HersheyFonts.HersheyTriplex, 0.5, Scalar.Black);
                    }
                }
            }
            using (new Window("died.tw", org))
            {
                Cv2.WaitKey();
            }
        }
    }
}
