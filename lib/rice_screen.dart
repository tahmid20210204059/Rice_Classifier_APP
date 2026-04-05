import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'rice_classifier.dart';

class RiceScreen extends StatefulWidget {
  const RiceScreen({super.key});
  @override
  State<RiceScreen> createState() => _RiceScreenState();
}

class _RiceScreenState extends State<RiceScreen> {
  final RiceClassifier _classifier = RiceClassifier();
  final ImagePicker _picker = ImagePicker();

  File? _imageFile;
  img.Image? _decodedImage;

  // Aspect ratio LOCKED = 1.61:1 (actual training data theke calculated)
  // User size change korte parbe, shape change hobe na
  // Scale difference ta preprocessing e 220px normalization handle korbe
  static const double _aspectRatio = 1.61;
  static const double _innerRatio = 0.80; // inner yellow = actual crop area

  double _boxHeight = 100.0; // user ei ta change korbe
  double get _boxWidth => _boxHeight * _aspectRatio; // auto maintain ratio

  Offset _rectCenter = Offset.zero;
  bool _isRectInitialized = false;

  // Display Calculations
  double _offsetX = 0;
  double _offsetY = 0;
  double _scale = 1.0;

  // Processed Images
  img.Image? _croppedAndCenteredImage;
  img.Image? _finalProcessedImage;

  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _classifier.loadModel();
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  Future<void> pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;

      setState(() => _isLoading = true);

      File file = File(pickedFile.path);
      final bytes = await file.readAsBytes();
      img.Image? decoded = img.decodeImage(bytes);

      if (decoded != null) {
        if (decoded.width > decoded.height && source == ImageSource.camera) {
          decoded = img.copyRotate(decoded, angle: 90);
        }
        decoded = img.bakeOrientation(decoded);

        setState(() {
          _imageFile = file;
          _decodedImage = decoded;
          _isRectInitialized = false;
          _croppedAndCenteredImage = null;
          _finalProcessedImage = null;
          _isLoading = false;
        });
      }
    } catch (e) {
      print("Image Pick Error: $e");
      setState(() => _isLoading = false);
    }
  }

  Future<void> cropAndPrepare() async {
    if (_decodedImage == null) return;

    setState(() => _isLoading = true);
    await Future.delayed(const Duration(milliseconds: 50));

    try {
      // 1. INNER BOX coordinates → actual image coordinates
      // Outer box = visual, Inner box = actual crop (80% of outer)
      double cropBoxW = _boxWidth * _innerRatio;
      double cropBoxH = _boxHeight * _innerRatio;

      double actualW = cropBoxW / _scale;
      double actualH = cropBoxH / _scale;
      double actualCx = (_rectCenter.dx - _offsetX) / _scale;
      double actualCy = (_rectCenter.dy - _offsetY) / _scale;

      int x = (actualCx - actualW / 2).toInt();
      int y = (actualCy - actualH / 2).toInt();
      int w = actualW.toInt();
      int h = actualH.toInt();

      x = math.max(0, x);
      y = math.max(0, y);
      if (x + w > _decodedImage!.width) w = _decodedImage!.width - x;
      if (y + h > _decodedImage!.height) h = _decodedImage!.height - y;

      // 2. CROP
      img.Image croppedRice = img.copyCrop(_decodedImage!, x: x, y: y, width: w, height: h);

      // 3. THRESHOLD 180 — background black
      for (final pixel in croppedRice) {
        double brightness = (pixel.r + pixel.g + pixel.b) / 3.0;
        if (brightness < 170) {
          pixel.r = 0;
          pixel.g = 0;
          pixel.b = 0;
        }
      }

      // 4. Black canvas 256x256
      int targetSize = 256;
      img.Image finalCanvas = img.Image(width: targetSize, height: targetSize);
      img.fill(finalCanvas, color: img.ColorRgb8(0, 0, 0));

      // 5. Resize — training e max_size=220
      const double maxRiceSize = 220.0;
      double scaleFactor = maxRiceSize / math.max(croppedRice.width, croppedRice.height);
      int newWidth = (croppedRice.width * scaleFactor).toInt();
      int newHeight = (croppedRice.height * scaleFactor).toInt();

      img.Image resizedRice = img.copyResize(
        croppedRice,
        width: newWidth,
        height: newHeight,
        interpolation: img.Interpolation.linear,
      );

      // 6. Center on canvas
      int dstX = (targetSize - newWidth) ~/ 2;
      int dstY = (targetSize - newHeight) ~/ 2;
      img.compositeImage(finalCanvas, resizedRice, dstX: dstX, dstY: dstY);

      setState(() {
        _croppedAndCenteredImage = finalCanvas;
        _finalProcessedImage = finalCanvas;
        _isLoading = false;
      });

    } catch (e) {
      setState(() => _isLoading = false);
      print("Crop Error: $e");
    }
  }

  Future<void> performAnalysis() async {
    if (_finalProcessedImage == null) return;

    setState(() => _isLoading = true);

    try {
      var result = await _classifier.predict(_finalProcessedImage!);
      setState(() => _isLoading = false);

      if (result.containsKey("success") && result["success"] == true) {
        showResultDialog(result["allResults"]);
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(result["error"] ?? "Unknown Error"),
              backgroundColor: Colors.redAccent,
            ),
          );
        }
      }
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error: $e"), backgroundColor: Colors.red),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text("Rice Doctor",
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
      ),
      body: Column(
        children: [
          Expanded(
            child: _imageFile == null
                ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.add_photo_alternate,
                      size: 80, color: Colors.grey[800]),
                  const SizedBox(height: 16),
                  const Text("Select a Rice Grain Image",
                      style: TextStyle(color: Colors.grey, fontSize: 16)),
                  const SizedBox(height: 8),
                  const Text("Camera or Gallery",
                      style: TextStyle(color: Colors.white24, fontSize: 13)),
                ],
              ),
            )
                : LayoutBuilder(
              builder: (context, constraints) {
                if (_decodedImage != null) {
                  double screenW = constraints.maxWidth;
                  double screenH = constraints.maxHeight;
                  double imgW = _decodedImage!.width.toDouble();
                  double imgH = _decodedImage!.height.toDouble();

                  double scaleX = screenW / imgW;
                  double scaleY = screenH / imgH;
                  _scale = math.min(scaleX, scaleY);

                  double displayW = imgW * _scale;
                  double displayH = imgH * _scale;
                  _offsetX = (screenW - displayW) / 2;
                  _offsetY = (screenH - displayH) / 2;
                }

                if (!_isRectInitialized && _decodedImage != null) {
                  _rectCenter = Offset(
                      constraints.maxWidth / 2,
                      constraints.maxHeight / 2);
                  _isRectInitialized = true;
                }

                return Stack(
                  children: [
                    Center(
                        child: Image.file(_imageFile!, fit: BoxFit.contain)),

                    if (_croppedAndCenteredImage == null) ...[
                      // Overlay + fixed box painter
                      CustomPaint(
                        size: Size(constraints.maxWidth, constraints.maxHeight),
                        painter: FixedBoxPainter(
                          center: _rectCenter,
                          boxWidth: _boxWidth,
                          boxHeight: _boxHeight,
                          innerRatio: _innerRatio,
                        ),
                      ),

                      // Draggable area (move only, resize via slider)
                      Positioned(
                        left: _rectCenter.dx - _boxWidth / 2,
                        top: _rectCenter.dy - _boxHeight / 2,
                        child: GestureDetector(
                          onPanUpdate: (details) {
                            setState(() {
                              _rectCenter = Offset(
                                _rectCenter.dx + details.delta.dx,
                                _rectCenter.dy + details.delta.dy,
                              );
                            });
                          },
                          child: Container(
                            width: _boxWidth,
                            height: _boxHeight,
                            color: Colors.transparent,
                            child: const Center(
                              child: Icon(Icons.open_with,
                                  color: Colors.white54, size: 18),
                            ),
                          ),
                        ),
                      ),

                      // Instruction — box er nichey
                      Positioned(
                        left: 0,
                        right: 0,
                        top: _rectCenter.dy + _boxHeight / 2 + 10,
                        child: const Center(
                          child: Text(
                            "Rice ta yellow box e fit korো",
                            style: TextStyle(
                              color: Colors.yellowAccent,
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              shadows: [
                                Shadow(color: Colors.black, blurRadius: 6),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ] else ...[
                      // Preview
                      Container(
                        color: Colors.black.withOpacity(0.92),
                        child: Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Text("MODEL INPUT PREVIEW",
                                  style: TextStyle(
                                      color: Colors.greenAccent,
                                      fontSize: 13,
                                      letterSpacing: 1.2)),
                              const SizedBox(height: 15),
                              Container(
                                width: 256,
                                height: 256,
                                decoration: BoxDecoration(
                                  border: Border.all(
                                      color: Colors.greenAccent, width: 2),
                                  boxShadow: [
                                    BoxShadow(
                                        color: Colors.greenAccent
                                            .withOpacity(0.2),
                                        blurRadius: 10)
                                  ],
                                ),
                                child: _finalProcessedImage != null
                                    ? Image.memory(
                                  Uint8List.fromList(img.encodeJpg(
                                      _finalProcessedImage!)),
                                  fit: BoxFit.contain,
                                  filterQuality: FilterQuality.high,
                                )
                                    : const CircularProgressIndicator(),
                              ),
                              const SizedBox(height: 10),
                              const Text(
                                  "256x256 • 220px Rice • Centered",
                                  style: TextStyle(
                                      color: Colors.white38,
                                      fontSize: 12)),
                            ],
                          ),
                        ),
                      ),
                      Positioned(
                        top: 20,
                        right: 20,
                        child: CircleAvatar(
                          backgroundColor: Colors.white24,
                          child: IconButton(
                            icon: const Icon(Icons.close,
                                color: Colors.white),
                            onPressed: () {
                              setState(() {
                                _croppedAndCenteredImage = null;
                                _finalProcessedImage = null;
                              });
                            },
                          ),
                        ),
                      ),
                    ],
                  ],
                );
              },
            ),
          ),

          // Bottom controls
          Container(
            padding: const EdgeInsets.only(
                top: 20, bottom: 30, left: 20, right: 20),
            decoration: BoxDecoration(
              color: Colors.grey[900],
              borderRadius:
              const BorderRadius.vertical(top: Radius.circular(20)),
              border:
              Border(top: BorderSide(color: Colors.white.withOpacity(0.1))),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (_imageFile != null) ...[
                  if (_croppedAndCenteredImage == null) ...[
                    // ✅ Size slider — aspect ratio 1.61:1 locked
                    Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text("Box Size",
                                  style: TextStyle(color: Colors.white70, fontSize: 12)),
                              Text(
                                "${_boxWidth.toInt()} × ${_boxHeight.toInt()}  (ratio locked 1.61:1)",
                                style: const TextStyle(
                                    color: Colors.white38, fontSize: 11),
                              ),
                            ],
                          ),
                          SliderTheme(
                            data: SliderTheme.of(context).copyWith(
                              trackHeight: 2,
                              thumbShape: const RoundSliderThumbShape(
                                  enabledThumbRadius: 8),
                              overlayShape: const RoundSliderOverlayShape(
                                  overlayRadius: 16),
                            ),
                            child: Slider(
                              value: _boxHeight,
                              min: 60,
                              max: 200,
                              activeColor: Colors.greenAccent,
                              inactiveColor: Colors.grey[700],
                              onChanged: (v) => setState(() => _boxHeight = v),
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.greenAccent,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10)),
                        ),
                        onPressed: _isLoading ? null : cropAndPrepare,
                        icon: const Icon(Icons.cut, color: Colors.black),
                        label: Text(
                          _isLoading ? "PROCESSING..." : "CROP & PREPARE",
                          style: const TextStyle(
                              color: Colors.black,
                              fontWeight: FontWeight.bold,
                              fontSize: 16),
                        ),
                      ),
                    ),
                  ] else ...[
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blueAccent,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10)),
                        ),
                        onPressed: _isLoading ? null : performAnalysis,
                        icon: _isLoading
                            ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                                color: Colors.white, strokeWidth: 2))
                            : const Icon(Icons.analytics_outlined),
                        label: Text(
                          _isLoading ? " ANALYZING..." : " START DIAGNOSIS",
                          style: const TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 16),
                        ),
                      ),
                    ),
                  ],
                  const SizedBox(height: 15),
                ],
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildImagePickerButton(Icons.camera_alt, "Camera",
                            () => pickImage(ImageSource.camera)),
                    _buildImagePickerButton(Icons.photo_library, "Gallery",
                            () => pickImage(ImageSource.gallery)),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImagePickerButton(
      IconData icon, String label, VoidCallback onPressed) {
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadius.circular(10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 12),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.white24),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          children: [
            Icon(icon, color: Colors.greenAccent, size: 28),
            const SizedBox(height: 5),
            Text(label,
                style:
                const TextStyle(color: Colors.white70, fontSize: 12)),
          ],
        ),
      ),
    );
  }

  void showResultDialog(List<dynamic> predictions) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Colors.grey[900],
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16)),
        title: Column(
          children: [
            const Icon(Icons.verified, color: Colors.greenAccent, size: 40),
            const SizedBox(height: 10),
            const Text("Analysis Result",
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold)),
            Divider(color: Colors.white.withOpacity(0.1)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: predictions
              .take(3)
              .map((p) => Container(
            margin: const EdgeInsets.only(bottom: 10),
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: Colors.grey[800],
              borderRadius: BorderRadius.circular(12),
              border: p == predictions.first
                  ? Border.all(
                  color: Colors.greenAccent, width: 1)
                  : null,
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    p["label"].toString().toUpperCase(),
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: p == predictions.first ? 16 : 14,
                    ),
                  ),
                ),
                Text(
                  "${(p["score"] * 100).toStringAsFixed(1)}%",
                  style: TextStyle(
                    color: p == predictions.first
                        ? Colors.greenAccent
                        : Colors.white70,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ))
              .toList(),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("CLOSE",
                style: TextStyle(color: Colors.white54)),
          ),
        ],
      ),
    );
  }
}

class FixedBoxPainter extends CustomPainter {
  final Offset center;
  final double boxWidth;
  final double boxHeight;
  final double innerRatio;

  FixedBoxPainter({
    required this.center,
    required this.boxWidth,
    required this.boxHeight,
    required this.innerRatio,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final boxRect = Rect.fromCenter(
        center: center, width: boxWidth, height: boxHeight);

    // Dark overlay with hole
    final overlayPaint = Paint()
      ..color = Colors.black.withOpacity(0.65);
    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRect(boxRect)
      ..fillType = PathFillType.evenOdd;
    canvas.drawPath(path, overlayPaint);

    // Green border — fixed box
    final greenPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawRect(boxRect, greenPaint);

    // Corner accents — visual guide
    final cornerPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    const double cornerLen = 12.0;

    // Top-left
    canvas.drawLine(boxRect.topLeft, boxRect.topLeft + const Offset(cornerLen, 0), cornerPaint);
    canvas.drawLine(boxRect.topLeft, boxRect.topLeft + const Offset(0, cornerLen), cornerPaint);
    // Top-right
    canvas.drawLine(boxRect.topRight, boxRect.topRight + const Offset(-cornerLen, 0), cornerPaint);
    canvas.drawLine(boxRect.topRight, boxRect.topRight + const Offset(0, cornerLen), cornerPaint);
    // Bottom-left
    canvas.drawLine(boxRect.bottomLeft, boxRect.bottomLeft + const Offset(cornerLen, 0), cornerPaint);
    canvas.drawLine(boxRect.bottomLeft, boxRect.bottomLeft + const Offset(0, -cornerLen), cornerPaint);
    // Bottom-right
    canvas.drawLine(boxRect.bottomRight, boxRect.bottomRight + const Offset(-cornerLen, 0), cornerPaint);
    canvas.drawLine(boxRect.bottomRight, boxRect.bottomRight + const Offset(0, -cornerLen), cornerPaint);

    // ✅ Inner yellow box — ACTUAL CROP AREA
    // Rice eta fill korbe — preprocessing e ei area theke crop hobe
    final yellowPaint = Paint()
      ..color = Colors.yellowAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    final innerRect = Rect.fromCenter(
      center: center,
      width: boxWidth * innerRatio,
      height: boxHeight * innerRatio,
    );
    canvas.drawRect(innerRect, yellowPaint);
  }

  @override
  bool shouldRepaint(covariant FixedBoxPainter oldDelegate) => true;
}