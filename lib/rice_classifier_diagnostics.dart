import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class RiceClassifierDiagnostics {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isModelLoaded = false;

  static const int inputSize = 256;
  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  String? _detectedFormat;
  List<int>? _detectedInputShape;

  // ==================== COMPLETE DIAGNOSTIC TEST ====================
  Future<void> runCompleteDiagnostics(img.Image testImage, String expectedLabel) async {
    print("\n" + "🔬"*50);
    print("COMPLETE RICE CLASSIFIER DIAGNOSTICS - NHWC VERSION");
    print("🔬"*50 + "\n");

    await _testModelLoading();
    await _testLabelLoading();
    await _testTensorFormats();
    await _testFormatDetection();
    _testInputImage(testImage);
    await _testNHWCConversionCorrectness(testImage);
    await _testSinglePrediction(testImage, "ORIGINAL");

    img.Image flippedH = img.flip(img.Image.from(testImage), direction: img.FlipDirection.horizontal);
    await _testSinglePrediction(flippedH, "HORIZONTAL FLIP");

    img.Image flippedV = img.flip(img.Image.from(testImage), direction: img.FlipDirection.vertical);
    await _testSinglePrediction(flippedV, "VERTICAL FLIP");

    img.Image rotated90 = img.copyRotate(testImage, angle: 90);
    await _testSinglePrediction(rotated90, "90° ROTATION");

    await _testPredictionConsistency(testImage);
    await _testExpectedLabel(expectedLabel);
    await _testConfidenceDistribution(testImage);
    _testRawPixelValues(testImage);
    await _testNormalizedValueRanges(testImage);

    print("\n" + "🔬"*50);
    print("DIAGNOSTICS COMPLETE");
    print("🔬"*50 + "\n");
  }

  Future<void> _testModelLoading() async {
    print("\n" + "="*70);
    print("TEST 1: MODEL LOADING");
    print("="*70);

    try {
      final options = InterpreterOptions();
      options.threads = 4;
      _interpreter = await Interpreter.fromAsset('assets/model.tflite', options: options);
      print("✅ Model loaded successfully");
      print("📊 Interpreter created: ${_interpreter != null}");
    } catch (e) {
      print("❌ Model loading FAILED: $e");
      return;
    }
  }

  Future<void> _testLabelLoading() async {
    print("\n" + "="*70);
    print("TEST 2: LABEL LOADING");
    print("="*70);

    try {
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').where((l) => l.isNotEmpty).map((e) => e.trim()).toList();
      print("✅ Labels loaded successfully");
      print("📊 Total labels: ${_labels.length}");
      print("\n📋 First 10 labels:");
      for (int i = 0; i < min(10, _labels.length); i++) {
        print("   ${i+1}. ${_labels[i]}");
      }
      if (_labels.length > 10) {
        print("   ...");
        print("   ${_labels.length}. ${_labels.last}");
      }
      _isModelLoaded = true;
    } catch (e) {
      print("❌ Label loading FAILED: $e");
    }
  }

  Future<void> _testTensorFormats() async {
    print("\n" + "="*70);
    print("TEST 3: TENSOR FORMAT & SHAPE ANALYSIS");
    print("="*70);
    if (_interpreter == null) {
      print("❌ Interpreter not loaded");
      return;
    }

    var inputTensor = _interpreter!.getInputTensor(0);
    print("\n📥 INPUT TENSOR:");
    print("   Shape: ${inputTensor.shape}");
    print("   Type: ${inputTensor.type}");

    _detectedInputShape = inputTensor.shape;

    var shape = inputTensor.shape;
    if (shape.length == 4) {
      if (shape[1] == 3) {
        print("   ✅ Detected: NCHW format");
        _detectedFormat = "NCHW";
      } else if (shape[3] == 3) {
        print("   ✅ Detected: NHWC format");
        _detectedFormat = "NHWC";
      }
    }

    var outputTensor = _interpreter!.getOutputTensor(0);
    print("\n📤 OUTPUT TENSOR:");
    print("   Shape: ${outputTensor.shape}");
    print("   Type: ${outputTensor.type}");
    print("   Num classes: ${outputTensor.shape.last}");

    if (outputTensor.shape.last == _labels.length) {
      print("   ✅ Output classes match label count");
    } else {
      print("   ⚠️  WARNING: Mismatch! Output=${outputTensor.shape.last}, Labels=${_labels.length}");
    }
  }

  Future<void> _testFormatDetection() async {
    print("\n" + "="*70);
    print("TEST 4: FORMAT VERIFICATION");
    print("="*70);

    if (_detectedFormat == "NHWC") {
      print("✅ Model uses NHWC format");
      print("✅ Code is using NHWC conversion");
      print("✅ FORMAT MATCH - Should work correctly!");
    } else {
      print("⚠️  Model uses: $_detectedFormat");
      print("⚠️  This is unexpected!");
    }
  }

  void _testInputImage(img.Image image) {
    print("\n" + "="*70);
    print("TEST 5: INPUT IMAGE ANALYSIS");
    print("="*70);
    print("📐 Original dimensions: ${image.width}x${image.height}");
    print("🎨 Number of channels: ${image.numChannels}");

    var center = image.getPixel(image.width ~/ 2, image.height ~/ 2);
    print("\n🎨 Center pixel: R=${center.r.toInt()}, G=${center.g.toInt()}, B=${center.b.toInt()}");

    int blackPixels = 0;
    int totalPixels = image.width * image.height;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        var pixel = image.getPixel(x, y);
        if (pixel.r < 10 && pixel.g < 10 && pixel.b < 10) blackPixels++;
      }
    }
    double blackPercentage = (blackPixels / totalPixels) * 100;
    print("\n🖤 Black background: ${blackPercentage.toStringAsFixed(2)}% of pixels");
  }

  Future<void> _testNHWCConversionCorrectness(img.Image image) async {
    print("\n" + "="*70);
    print("TEST 6: NHWC CONVERSION VERIFICATION");
    print("="*70);

    img.Image processed = image.width == inputSize && image.height == inputSize
        ? image
        : img.copyResize(image, width: inputSize, height: inputSize);

    var inputBuffer = _imageToByteListFloat32_NHWC(processed);

    // Check if it's Float32List
    if (inputBuffer is Float32List) {
      print("❌ ERROR: Returned Float32List instead of reshaped tensor");
      print("   This will cause issues!");
    } else {
      print("✅ Conversion returned properly reshaped tensor");
    }

    // Verify first few values
    print("\n🔍 Sample converted values (first pixel):");
    print("   This should show R, G, B normalized values");

    var pixel = processed.getPixel(0, 0);
    double expectedR = ((pixel.r / 255.0) - mean[0]) / std[0];
    double expectedG = ((pixel.g / 255.0) - mean[1]) / std[1];
    double expectedB = ((pixel.b / 255.0) - mean[2]) / std[2];

    print("   Expected R: ${expectedR.toStringAsFixed(4)}");
    print("   Expected G: ${expectedG.toStringAsFixed(4)}");
    print("   Expected B: ${expectedB.toStringAsFixed(4)}");
  }

  Future<void> _testSinglePrediction(img.Image image, String testName) async {
    print("\n" + "="*70);
    print("TEST: PREDICTION - $testName");
    print("="*70);
    if (_interpreter == null || !_isModelLoaded) {
      print("❌ Model not ready");
      return;
    }

    try {
      img.Image processed = image.width == inputSize && image.height == inputSize
          ? image
          : img.copyResize(image, width: inputSize, height: inputSize);

      var inputBuffer = _imageToByteListFloat32_NHWC(processed);
      var outputTensor = _interpreter!.getOutputTensor(0);
      var numClasses = outputTensor.shape.last;
      var outputBuffer = List.filled(numClasses, 0.0).reshape([1, numClasses]);

      print("🔄 Running inference...");
      _interpreter!.run(inputBuffer, outputBuffer);

      List<double> logits = List<double>.from(outputBuffer[0]);
      List<double> probs = _softmax(logits);

      double maxProb = probs.reduce(max);
      print("📈 Max probability: ${(maxProb * 100).toStringAsFixed(2)}%");

      List<Map<String, dynamic>> results = [];
      for (int i = 0; i < min(_labels.length, numClasses); i++) {
        results.add({"label": _labels[i], "score": probs[i]});
      }
      results.sort((a, b) => b["score"].compareTo(a["score"]));

      print("\n🏆 TOP 10 PREDICTIONS:");
      for (int i = 0; i < min(10, results.length); i++) {
        String label = results[i]["label"];
        double score = results[i]["score"] * 100;
        print("   ${(i + 1).toString().padLeft(2)}. ${label.padRight(20)} ${score.toStringAsFixed(2).padLeft(6)}%");
      }

      // Check if predictions are uniform (model not working)
      double avgProb = probs.reduce((a, b) => a + b) / probs.length;
      double variance = probs.map((p) => pow(p - avgProb, 2)).reduce((a, b) => a + b) / probs.length;
      double stdDev = sqrt(variance);

      if (stdDev < 0.01) {
        print("\n⚠️  WARNING: Predictions are too uniform!");
        print("   Model might not be working correctly");
        print("   StdDev: ${(stdDev * 100).toStringAsFixed(4)}%");
      } else {
        print("\n✅ Good prediction variance (StdDev: ${(stdDev * 100).toStringAsFixed(2)}%)");
      }

    } catch (e, stack) {
      print("❌ Prediction FAILED: $e");
      print("Stack trace: $stack");
    }
  }

  Future<void> _testPredictionConsistency(img.Image image) async {
    print("\n" + "="*70);
    print("TEST 7: PREDICTION CONSISTENCY (5 runs)");
    print("="*70);
    if (_interpreter == null || !_isModelLoaded) {
      print("❌ Model not ready");
      return;
    }

    List<String> predictions = [];
    for (int run = 1; run <= 5; run++) {
      try {
        img.Image processed = image.width == inputSize && image.height == inputSize
            ? image
            : img.copyResize(image, width: inputSize, height: inputSize);

        var inputBuffer = _imageToByteListFloat32_NHWC(processed);
        var outputTensor = _interpreter!.getOutputTensor(0);
        var numClasses = outputTensor.shape.last;
        var outputBuffer = List.filled(numClasses, 0.0).reshape([1, numClasses]);

        _interpreter!.run(inputBuffer, outputBuffer);
        List<double> probs = _softmax(List<double>.from(outputBuffer[0]));

        int maxIdx = 0;
        double maxProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
          if (probs[i] > maxProb) {
            maxProb = probs[i];
            maxIdx = i;
          }
        }
        predictions.add(_labels[maxIdx]);
        print("Run $run: ${_labels[maxIdx]} (${(maxProb * 100).toStringAsFixed(2)}%)");
      } catch (e) {
        print("Run $run: FAILED - $e");
      }
    }

    bool allSame = predictions.every((p) => p == predictions[0]);
    if (allSame) {
      print("\n✅ CONSISTENT: All predictions identical");
    } else {
      print("\n❌ INCONSISTENT: Different predictions!");
    }
  }

  Future<void> _testExpectedLabel(String expectedLabel) async {
    print("\n" + "="*70);
    print("TEST 8: EXPECTED LABEL ANALYSIS");
    print("="*70);
    print("🎯 Expected label: '$expectedLabel'");

    if (_labels.contains(expectedLabel)) {
      int index = _labels.indexOf(expectedLabel);
      print("✅ Label exists in model at index: $index");
    } else {
      print("⚠️  Expected label NOT FOUND in label list");
      print("\n🔍 Searching for similar labels:");
      for (var label in _labels) {
        if (label.toLowerCase().contains(expectedLabel.toLowerCase())) {
          print("   Similar: $label");
        }
      }
    }
  }

  Future<void> _testConfidenceDistribution(img.Image image) async {
    print("\n" + "="*70);
    print("TEST 9: CONFIDENCE DISTRIBUTION");
    print("="*70);
    if (_interpreter == null || !_isModelLoaded) {
      print("❌ Model not ready");
      return;
    }

    try {
      img.Image processed = image.width == inputSize && image.height == inputSize
          ? image
          : img.copyResize(image, width: inputSize, height: inputSize);

      var inputBuffer = _imageToByteListFloat32_NHWC(processed);
      var outputTensor = _interpreter!.getOutputTensor(0);
      var numClasses = outputTensor.shape.last;
      var outputBuffer = List.filled(numClasses, 0.0).reshape([1, numClasses]);

      _interpreter!.run(inputBuffer, outputBuffer);
      List<double> probs = _softmax(List<double>.from(outputBuffer[0]));

      double mean = probs.reduce((a, b) => a + b) / probs.length;
      double variance = probs.map((p) => pow(p - mean, 2)).reduce((a, b) => a + b) / probs.length;
      double stdDev = sqrt(variance);

      print("📊 Mean probability: ${(mean * 100).toStringAsFixed(4)}%");
      print("📊 Std Dev: ${(stdDev * 100).toStringAsFixed(4)}%");
      print("📊 Min probability: ${(probs.reduce(min) * 100).toStringAsFixed(4)}%");
      print("📊 Max probability: ${(probs.reduce(max) * 100).toStringAsFixed(4)}%");

      if (stdDev < 0.01) {
        print("\n❌ CRITICAL: Uniform distribution - model not discriminating!");
      } else {
        print("\n✅ Good variance - model is discriminating between classes");
      }
    } catch (e) {
      print("❌ Test FAILED: $e");
    }
  }

  void _testRawPixelValues(img.Image image) {
    print("\n" + "="*70);
    print("TEST 10: RAW PIXEL ANALYSIS (Sample of 100 pixels)");
    print("="*70);

    img.Image processed = image.width == inputSize && image.height == inputSize
        ? image
        : img.copyResize(image, width: inputSize, height: inputSize);

    List<int> rValues = [];
    List<int> gValues = [];
    List<int> bValues = [];

    Random rand = Random();
    for (int i = 0; i < 100; i++) {
      int x = rand.nextInt(inputSize);
      int y = rand.nextInt(inputSize);
      var pixel = processed.getPixel(x, y);
      rValues.add(pixel.r.toInt());
      gValues.add(pixel.g.toInt());
      bValues.add(pixel.b.toInt());
    }

    double avgR = rValues.reduce((a, b) => a + b) / rValues.length;
    double avgG = gValues.reduce((a, b) => a + b) / gValues.length;
    double avgB = bValues.reduce((a, b) => a + b) / bValues.length;

    print("🔴 Red avg: ${avgR.toStringAsFixed(2)} (range: 0-255)");
    print("🟢 Green avg: ${avgG.toStringAsFixed(2)} (range: 0-255)");
    print("🔵 Blue avg: ${avgB.toStringAsFixed(2)} (range: 0-255)");
  }

  Future<void> _testNormalizedValueRanges(img.Image image) async {
    print("\n" + "="*70);
    print("TEST 11: NORMALIZED VALUE ANALYSIS");
    print("="*70);

    img.Image processed = image.width == inputSize && image.height == inputSize
        ? image
        : img.copyResize(image, width: inputSize, height: inputSize);

    List<double> rNorm = [];
    List<double> gNorm = [];
    List<double> bNorm = [];

    for (int i = 0; i < 1000; i++) {
      Random rand = Random();
      int x = rand.nextInt(inputSize);
      int y = rand.nextInt(inputSize);
      var pixel = processed.getPixel(x, y);

      rNorm.add((pixel.r / 255.0 - mean[0]) / std[0]);
      gNorm.add((pixel.g / 255.0 - mean[1]) / std[1]);
      bNorm.add((pixel.b / 255.0 - mean[2]) / std[2]);
    }

    double avgR = rNorm.reduce((a, b) => a + b) / rNorm.length;
    double avgG = gNorm.reduce((a, b) => a + b) / gNorm.length;
    double avgB = bNorm.reduce((a, b) => a + b) / bNorm.length;

    print("🔴 Red normalized avg: ${avgR.toStringAsFixed(4)}");
    print("🟢 Green normalized avg: ${avgG.toStringAsFixed(4)}");
    print("🔵 Blue normalized avg: ${avgB.toStringAsFixed(4)}");
    print("\n📊 Using ImageNet normalization:");
    print("   mean = $mean");
    print("   std = $std");

    // Typical normalized range for most images is -2.5 to +2.5
    if (avgR < -3.0 || avgG < -3.0 || avgB < -3.0) {
      print("\n⚠️  WARNING: Normalized values seem too negative");
      print("   Image might be very dark (black background)");
    }
  }

  // NHWC Conversion Function
  Object _imageToByteListFloat32_NHWC(img.Image image) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        var pixel = image.getPixel(x, y);

        // NHWC order: R, G, B consecutively
        convertedBytes[pixelIndex++] = ((pixel.r / 255.0) - mean[0]) / std[0];
        convertedBytes[pixelIndex++] = ((pixel.g / 255.0) - mean[1]) / std[1];
        convertedBytes[pixelIndex++] = ((pixel.b / 255.0) - mean[2]) / std[2];
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
}