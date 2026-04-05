import 'package:flutter/material.dart';
import 'rice_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Rice Doctor',
      theme: ThemeData(
        brightness: Brightness.dark, // Dark theme default kora holo demo er moto
        colorScheme: ColorScheme.dark(
          primary: Colors.greenAccent,
          surface: Colors.grey[900]!,
        ),
        useMaterial3: true,
      ),
      home: const RiceScreen(),
    );
  }
}