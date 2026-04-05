import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class RiceClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isModelLoaded = false;

  // 📏 Input Size
  static const int inputSize = 256;

  // 🧮 Normalization Statistics (Your confirmed values)
  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  Future<void> loadModel() async {
    try {
      // 🚀 4 Threads for faster processing
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset('assets/rice_model_new.tflite', options: options);

      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').where((l) => l.isNotEmpty).map((e) => e.trim()).toList();

      _isModelLoaded = true;
      print("✅ Ultra 8-Fold TTA Model Loaded");
    } catch (e) {
      print("❌ Error loading model: $e");
    }
  }

  /// 🔥 ULTRA 8-FOLD TTA PREDICTION
  Future<Map<String, dynamic>> predict(img.Image originalImage) async {
    if (!_isModelLoaded) return {"error": "Model not loaded"};

    print("🔄 Starting 8-Fold TTA Analysis...");

    // ১. ৮টি আলাদা ইমেজ তৈরি করা (Geometric Transformations)
    List<img.Image> ttaImages = [];

    // --- GROUP A: Rotations ---
    ttaImages.add(originalImage);                                // 1. Original
    ttaImages.add(img.copyRotate(originalImage, angle: 90));     // 2. 90 Deg
    ttaImages.add(img.copyRotate(originalImage, angle: 180));    // 3. 180 Deg
    ttaImages.add(img.copyRotate(originalImage, angle: 270));    // 4. 270 Deg

    // --- GROUP B: Flips (Mirroring) ---
    img.Image flippedH = img.copyFlip(originalImage, direction: img.FlipDirection.horizontal);
    ttaImages.add(flippedH);                                     // 5. Flip Horizontal

    img.Image flippedV = img.copyFlip(originalImage, direction: img.FlipDirection.vertical);
    ttaImages.add(flippedV);                                     // 6. Flip Vertical

    // --- GROUP C: Flip + Rotate ---
    ttaImages.add(img.copyRotate(flippedH, angle: 90));          // 7. Flip H + Rot 90
    ttaImages.add(img.copyRotate(flippedV, angle: 90));          // 8. Flip V + Rot 90

    // স্কোর জমা রাখার বাফার
    List<double> totalProbabilities = List.filled(_labels.length, 0.0);

    // ২. লুপ চালিয়ে ৮ বার প্রেডিকশন
    int count = 0;
    for (var image in ttaImages) {
      count++;

      // ইনপুট তৈরি
      var inputTensor = _imageToByteListFloat32(image);
      var outputTensor = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);

      // রান
      _interpreter!.run(inputTensor, outputTensor);

      // এই ইমেজের স্কোর
      List<double> logits = List<double>.from(outputTensor[0]);
      List<double> probs = _softmax(logits);

      // ডিবাগ প্রিন্ট (কনসোল চেক করুন কোন অ্যাঙ্গেলে কী রেজাল্ট আসছে)
      // print("📸 TTA Step $count: ${probs}");

      // টোটাল স্কোরের সাথে যোগ
      for (int i = 0; i < _labels.length; i++) {
        totalProbabilities[i] += probs[i];
      }
    }

    // ৩. গড় (Average) করা: সব যোগফলকে ৮ দিয়ে ভাগ
    List<Map<String, dynamic>> results = [];
    for (int i = 0; i < _labels.length; i++) {
      double avgScore = totalProbabilities[i] / ttaImages.length;
      results.add({"label": _labels[i], "score": avgScore});
    }

    // রেজাল্ট সর্ট
    results.sort((a, b) => b["score"].compareTo(a["score"]));

    // ডিবাগ: ফাইনাল রেজাল্ট প্রিন্ট
    print("🏆 Final Result: ${results.first['label']} (${(results.first['score']*100).toStringAsFixed(1)}%)");

    return {"success": true, "allResults": results};
  }

  /// 🛠️ PIXEL CONVERSION (NO RESIZE - Strictly Math)
  Object _imageToByteListFloat32(img.Image image) {
    // RiceScreen.dart থেকে 256x256 আসছে, তাই কোনো রিসাইজ দরকার নেই।

    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var y = 0; y < inputSize; y++) {
      for (var x = 0; x < inputSize; x++) {
        var pixel = image.getPixel(x, y);

        // 🧮 Formula: (Value/255 - Mean) / Std
        buffer[pixelIndex++] = ((pixel.r / 255.0) - mean[0]) / std[0]; // R
        buffer[pixelIndex++] = ((pixel.g / 255.0) - mean[1]) / std[1]; // G
        buffer[pixelIndex++] = ((pixel.b / 255.0) - mean[2]) / std[2]; // B
      }
    }

    return convertedBytes.reshape([1, inputSize, inputSize, 3]);
  }

  List<double> _softmax(List<double> logits) {
    if (logits.isEmpty) return [];
    double maxLogit = logits.reduce(max);
    List<double> exps = logits.map((x) => exp(x - maxLogit)).toList();
    double sumExps = exps.reduce((a, b) => a + b);
    return exps.map((x) => x / sumExps).toList();
  }

  void dispose() {
    _interpreter?.close();
  }
}