package org.tensorflow.codelabs.objectdetection

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity(), View.OnClickListener {
    companion object {
        const val TAG = "TFLite - ODT"
        const val REQUEST_IMAGE_CAPTURE: Int = 1
        private const val MAX_FONT_SIZE = 96F
        private const val INPUT_SIZE = 320  // Model expects 320x320, not 640x640!
        private const val NUM_CLASSES = 6
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
    }

    private lateinit var captureImageFab: Button
    private lateinit var inputImageView: ImageView
    private lateinit var imgSampleOne: ImageView
    private lateinit var imgSampleTwo: ImageView
    private lateinit var imgSampleThree: ImageView
    private lateinit var tvPlaceholder: TextView
    private lateinit var currentPhotoPath: String

    private var interpreter: Interpreter? = null
    private val labels = listOf("10", "100", "20", "200", "50", "500")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        captureImageFab = findViewById(R.id.captureImageFab)
        inputImageView = findViewById(R.id.imageView)
        imgSampleOne = findViewById(R.id.imgSampleOne)
        imgSampleTwo = findViewById(R.id.imgSampleTwo)
        imgSampleThree = findViewById(R.id.imgSampleThree)
        tvPlaceholder = findViewById(R.id.tvPlaceholder)

        captureImageFab.setOnClickListener(this)
        imgSampleOne.setOnClickListener(this)
        imgSampleTwo.setOnClickListener(this)
        imgSampleThree.setOnClickListener(this)

        // Load model
        try {
            val modelFile = FileUtil.loadMappedFile(this, "best22_2_26_float32.tflite")
            val options = Interpreter.Options()
            options.setNumThreads(4)
            interpreter = Interpreter(modelFile, options)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            setViewAndDetect(getCapturedImage())
        }
    }

    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.captureImageFab -> {
                try {
                    dispatchTakePictureIntent()
                } catch (e: ActivityNotFoundException) {
                    Log.e(TAG, e.message.toString())
                    Toast.makeText(this, "No camera app found", Toast.LENGTH_SHORT).show()
                }
            }
            R.id.imgSampleOne -> {
                setViewAndDetect(getSampleImage(R.drawable.img_meal_one))
            }
            R.id.imgSampleTwo -> {
                setViewAndDetect(getSampleImage(R.drawable.img_meal_two))
            }
            R.id.imgSampleThree -> {
                setViewAndDetect(getSampleImage(R.drawable.img_meal_three))
            }
        }
    }

    private fun runObjectDetection(bitmap: Bitmap) {
        try {
            Log.d(TAG, "=== Starting Currency Detection ===")
            if (interpreter == null) {
                Log.e(TAG, "Interpreter not initialized")
                return
            }

            // Log model details
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            Log.d(TAG, "Input tensor: shape=${inputTensor.shape().contentToString()}, dtype=${inputTensor.dataType()}")
            Log.d(TAG, "Output tensor: shape=${outputTensor.shape().contentToString()}, dtype=${outputTensor.dataType()}")

            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val inputBuffer = preprocessImage(resizedBitmap)

            // Verify buffer size
            val expectedBytes = inputTensor.numBytes()
            Log.d(TAG, "Buffer size: expected=$expectedBytes, actual=${inputBuffer.capacity()}")
            if (inputBuffer.capacity() != expectedBytes) {
                Log.e(TAG, "Buffer size mismatch!")
                return
            }

            // Log some input values
            inputBuffer.rewind()
            val sampleBytes = ByteArray(30)
            inputBuffer.get(sampleBytes)
            inputBuffer.rewind()
            Log.d(TAG, "First 30 input bytes: ${sampleBytes.take(30).joinToString(",") { (it.toInt() and 0xFF).toString() }}")

            // Output buffer
            val output = Array(1) { Array(10) { FloatArray(2100) } }

            Log.d(TAG, "Running inference...")
            interpreter!!.run(inputBuffer, output)
            Log.d(TAG, "Inference complete")

            // Analyze output in detail
            val outputArray = output[0]

            // Check if output is all zeros
            var hasNonZero = false
            var sumValues = 0.0
            var countNonZero = 0

            for (i in 0 until 2100) {
                for (c in 0 until 10) {
                    val v = outputArray[c][i]
                    sumValues += Math.abs(v.toDouble())
                    if (v != 0f) {
                        hasNonZero = true
                        countNonZero++
                    }
                }
            }

            Log.d(TAG, "Output analysis: hasNonZero=$hasNonZero, countNonZero=$countNonZero, sumAbs=$sumValues")

            // Log a few sample outputs
            Log.d(TAG, "Sample outputs at indices 0, 100, 500:")
            for (idx in listOf(0, 100, 500)) {
                Log.d(TAG, "  [$idx]: x=${outputArray[0][idx]}, y=${outputArray[1][idx]}, w=${outputArray[2][idx]}, h=${outputArray[3][idx]}")
                Log.d(TAG, "       classes: ${(4 until 10).map { outputArray[it][idx] }.joinToString(",")}")
            }

            // Find max value
            var maxVal = -Float.MAX_VALUE
            var maxIdx = -1
            var maxChannel = -1
            for (i in 0 until 2100) {
                for (c in 0 until 10) {
                    val v = outputArray[c][i]
                    if (v > maxVal) {
                        maxVal = v
                        maxIdx = i
                        maxChannel = c
                    }
                }
            }
            Log.d(TAG, "Max output: value=$maxVal at index=$maxIdx, channel=$maxChannel")

            if (maxIdx != -1) {
                Log.d(TAG, "Sample at max index $maxIdx:")
                Log.d(TAG, "  x=${outputArray[0][maxIdx]}, y=${outputArray[1][maxIdx]}, w=${outputArray[2][maxIdx]}, h=${outputArray[3][maxIdx]}")
                for (c in 0 until NUM_CLASSES) {
                    Log.d(TAG, "  class ${labels[c]}: ${outputArray[4+c][maxIdx]}")
                }
            }

            // Parse output (same as before)
            val detections = parseOutput(output[0], bitmap.width, bitmap.height)

            Log.d(TAG, "After NMS: ${detections.size} detections")

            // Convert to display format
            val resultToDisplay = detections.map { detection ->
                val text = String.format("₹%s (%.0f%%)",
                    detection.label,
                    detection.confidence * 100)
                DetectionResult(detection.bbox, text)
            }

            // Draw results
            val imgWithResult = drawDetectionResult(bitmap, resultToDisplay)

            runOnUiThread {
                inputImageView.setImageBitmap(imgWithResult)
                tvPlaceholder.visibility = View.INVISIBLE

                if (detections.isEmpty()) {
                    Toast.makeText(this, "No currency detected", Toast.LENGTH_SHORT).show()
                } else {
                    val totalValue = detections.sumOf { it.label.toIntOrNull() ?: 0 }
                    Toast.makeText(
                        this,
                        "Found ${detections.size} note(s), Total: ₹$totalValue",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Detection error", e)
            e.printStackTrace()
            runOnUiThread {
                Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Model expects FLOAT32 input [1, 320, 320, 3] in RGB order
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Android Bitmap.getPixels() returns ARGB in an int
        // We need RGB in float32 normalized to [0, 1]
        for (pixel in intValues) {
            // Extract RGB channels from ARGB int
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Put in RGB order (not BGR!)
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        return inputBuffer
    }

    private fun parseOutput(
        output: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numBoxes = output[0].size // 2100

        for (i in 0 until numBoxes) {
            // Coordinates (first 4 rows)
            val xCenter = output[0][i]
            val yCenter = output[1][i]
            val width = output[2][i]
            val height = output[3][i]

            // Find best class
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
                // YOLO outputs are normalized to [0, 1] range
                // Need to multiply by original image dimensions
                val left = (xCenter - width / 2) * originalWidth
                val top = (yCenter - height / 2) * originalHeight
                val right = (xCenter + width / 2) * originalWidth
                val bottom = (yCenter + height / 2) * originalHeight

                val bbox = RectF(left, top, right, bottom)

                // Debug first detection
                if (detections.isEmpty()) {
                    Log.d(TAG, "First detection bbox: [$left, $top, $right, $bottom]")
                    Log.d(TAG, "  Raw: xCenter=$xCenter, yCenter=$yCenter, w=$width, h=$height")
                    Log.d(TAG, "  Image size: $originalWidth x $originalHeight")
                }

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

    private fun setViewAndDetect(bitmap: Bitmap) {
        inputImageView.setImageBitmap(bitmap)
        tvPlaceholder.visibility = View.INVISIBLE
        lifecycleScope.launch(Dispatchers.Default) { runObjectDetection(bitmap) }
    }

    // Camera and image handling (same as before)
    private fun getCapturedImage(): Bitmap {
        val targetW = inputImageView.width
        val targetH = inputImageView.height
        val bmOptions = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
            BitmapFactory.decodeFile(currentPhotoPath, this)
            val photoW = outWidth
            val photoH = outHeight
            val scaleFactor = max(1, min(photoW / targetW, photoH / targetH))
            inJustDecodeBounds = false
            inSampleSize = scaleFactor
            inMutable = true
        }
        val exif = ExifInterface(currentPhotoPath)
        val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)
        val bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> rotateImage(bitmap, 90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> rotateImage(bitmap, 180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> rotateImage(bitmap, 270f)
            else -> bitmap
        }
    }

    private fun getSampleImage(drawable: Int): Bitmap =
        BitmapFactory.decodeResource(resources, drawable, BitmapFactory.Options().apply { inMutable = true })

    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { intent ->
            intent.resolveActivity(packageManager)?.also {
                val photoFile = try { createImageFile() } catch (e: IOException) {
                    Log.e(TAG, e.message.toString())
                    Toast.makeText(this, "Failed to create image file", Toast.LENGTH_SHORT).show()
                    null
                }
                photoFile?.also { file ->
                    val uri = FileProvider.getUriForFile(this, "org.tensorflow.codelabs.objectdetection.fileprovider", file)
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, uri)
                    startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
                }
            } ?: run {
                Log.e(TAG, "No camera activity found")
                Toast.makeText(this, "No camera app found", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun drawDetectionResult(bitmap: Bitmap, results: List<DetectionResult>): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint().apply { textAlign = Paint.Align.LEFT }

        for (r in results) {
            // Draw bounding box
            paint.color = Color.GREEN
            paint.strokeWidth = 8f
            paint.style = Paint.Style.STROKE
            canvas.drawRect(r.boundingBox, paint)

            // Draw text background
            val bounds = Rect()
            paint.style = Paint.Style.FILL
            paint.color = Color.BLACK
            paint.textSize = 48f
            paint.getTextBounds(r.text, 0, r.text.length, bounds)
            canvas.drawRect(
                r.boundingBox.left,
                r.boundingBox.top - bounds.height() - 10,
                r.boundingBox.left + bounds.width() + 20,
                r.boundingBox.top,
                paint
            )

            // Draw text
            paint.color = Color.WHITE
            canvas.drawText(r.text, r.boundingBox.left + 10, r.boundingBox.top - 10, paint)
        }
        return output
    }
}

data class DetectionResult(val boundingBox: RectF, val text: String)
data class Detection(val bbox: RectF, val label: String, val confidence: Float)