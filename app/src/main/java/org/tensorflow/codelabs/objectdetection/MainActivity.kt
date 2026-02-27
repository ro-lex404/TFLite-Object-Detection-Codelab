package org.tensorflow.codelabs.objectdetection

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/**
 * Real-time Currency Detection with CameraX
 * Detects Indian currency notes in real-time using the camera
 */
class MainActivity : AppCompatActivity() {
    companion object {
        const val TAG = "CurrencyDetector"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val INPUT_SIZE = 320
        private const val NUM_CLASSES = 6
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.45f
    }

    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var btnToggleCamera: Button
    private lateinit var tvStatus: TextView

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var isCameraRunning = false

    // Model
    private var interpreter: Interpreter? = null
    private val labels = listOf("10", "100", "20", "200", "50", "500")

    // Detection state
    private var lastDetectionTime = 0L
    private val detectionInterval = 100L // Run detection every 100ms

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        // Initialize UI
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        btnToggleCamera = findViewById(R.id.btnToggleCamera)
        tvStatus = findViewById(R.id.tvStatus)

        btnToggleCamera.setOnClickListener {
            if (isCameraRunning) {
                stopCamera()
            } else {
                startCamera()
            }
        }

        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Load model
        try {
            val modelFile = FileUtil.loadMappedFile(this, "currency_model.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelFile, options)
            Log.d(TAG, "✅ Model loaded successfully")
            tvStatus.text = "Model loaded. Tap Start to begin detection."
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error loading model", e)
            tvStatus.text = "Error loading model"
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_LONG).show()
        }

        // Request camera permissions
        if (allPermissionsGranted()) {
            tvStatus.text = "Ready. Tap Start Camera."
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun startCamera() {
        if (!allPermissionsGranted()) {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
            isCameraRunning = true
            btnToggleCamera.text = "Stop Camera"
            tvStatus.text = "Detecting..."
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        isCameraRunning = false
        btnToggleCamera.text = "Start Camera"
        tvStatus.text = "Camera stopped"
        overlayView.clear()
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        // Preview use case
        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Image analysis use case
        imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                    processImage(imageProxy)
                }
            }

        // Select back camera
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )
        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processImage(imageProxy: ImageProxy) {
        // Throttle detection to avoid overwhelming the system
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastDetectionTime < detectionInterval) {
            imageProxy.close()
            return
        }
        lastDetectionTime = currentTime

        try {
            val bitmap = imageProxy.toBitmap()
            val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)

            // Run detection
            val detections = detectObjects(rotatedBitmap)

            // Update overlay on UI thread
            runOnUiThread {
                overlayView.setDetections(detections, rotatedBitmap.width, rotatedBitmap.height)

                if (detections.isNotEmpty()) {
                    val total = detections.sumOf { it.label.toIntOrNull() ?: 0 }
                    tvStatus.text = "Detected: ${detections.size} note(s), Total: ₹$total"
                } else {
                    tvStatus.text = "No currency detected"
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Detection error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun detectObjects(bitmap: Bitmap): List<Detection> {
        if (interpreter == null) return emptyList()

        try {
            // Resize and preprocess
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
            val inputBuffer = preprocessImage(resizedBitmap)

            // Run inference
            val output = Array(1) { Array(10) { FloatArray(2100) } }
            interpreter!!.run(inputBuffer, output)

            // Parse and return detections
            return parseOutput(output[0], bitmap.width, bitmap.height)
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed", e)
            return emptyList()
        }
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixel in intValues) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        return inputBuffer
    }

    private fun parseOutput(output: Array<FloatArray>, imageWidth: Int, imageHeight: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numBoxes = output[0].size

        for (i in 0 until numBoxes) {
            val xCenter = output[0][i]
            val yCenter = output[1][i]
            val width = output[2][i]
            val height = output[3][i]

            var bestScore = 0f
            var bestClass = 0
            for (c in 0 until NUM_CLASSES) {
                val score = output[4 + c][i]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }

            if (bestScore >= CONFIDENCE_THRESHOLD) {
                // YOLO coordinates are normalized [0, 1]
                val left = (xCenter - width / 2) * imageWidth
                val top = (yCenter - height / 2) * imageHeight
                val right = (xCenter + width / 2) * imageWidth
                val bottom = (yCenter + height / 2) * imageHeight

                val bbox = RectF(left, top, right, bottom)
                detections.add(Detection(bbox, labels[bestClass], bestScore))
            }
        }

        return nonMaximumSuppression(detections)
    }

    private fun nonMaximumSuppression(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            selected.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (calculateIoU(sorted[i].bbox, sorted[j].bbox) > IOU_THRESHOLD) {
                    suppressed[j] = true
                }
            }
        }
        return selected
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)

        if (intersectionRight < intersectionLeft || intersectionBottom < intersectionTop) return 0f

        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val union = box1Area + box2Area - intersectionArea
        return intersectionArea / union
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                tvStatus.text = "Ready. Tap Start Camera."
            } else {
                Toast.makeText(this, "Permissions not granted", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor?.shutdown()
        interpreter?.close()
    }
}

/**
 * Custom view for drawing detection overlays
 */
class OverlayView @JvmOverloads constructor(
    context: android.content.Context,
    attrs: android.util.AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections = listOf<Detection>()
    private var imageWidth = 0
    private var imageHeight = 0

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 48f
        style = Paint.Style.FILL
    }

    private val backgroundPaint = Paint().apply {
        color = Color.BLACK
        alpha = 180
        style = Paint.Style.FILL
    }

    fun setDetections(newDetections: List<Detection>, imgWidth: Int, imgHeight: Int) {
        detections = newDetections
        imageWidth = imgWidth
        imageHeight = imgHeight
        invalidate()
    }

    fun clear() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (detections.isEmpty() || imageWidth == 0 || imageHeight == 0) return

        // Calculate scale factor to fit overlay on preview
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight

        detections.forEach { detection ->
            // Scale bounding box to overlay coordinates
            val box = RectF(
                detection.bbox.left * scaleX,
                detection.bbox.top * scaleY,
                detection.bbox.right * scaleX,
                detection.bbox.bottom * scaleY
            )

            // Draw bounding box
            canvas.drawRect(box, boxPaint)

            // Draw label
            val text = "₹${detection.label} (${(detection.confidence * 100).toInt()}%)"
            val textBounds = Rect()
            textPaint.getTextBounds(text, 0, text.length, textBounds)

            // Draw background for text
            canvas.drawRect(
                box.left,
                box.top - textBounds.height() - 20,
                box.left + textBounds.width() + 20,
                box.top,
                backgroundPaint
            )

            // Draw text
            canvas.drawText(text, box.left + 10, box.top - 10, textPaint)
        }
    }
}

data class Detection(val bbox: RectF, val label: String, val confidence: Float)