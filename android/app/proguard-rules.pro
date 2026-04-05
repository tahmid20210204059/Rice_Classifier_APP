# TensorFlow Lite ProGuard Rules
-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**