package org.tensorflow.codelabs.objectdetection

import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.*

class WeatherActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private var tts: TextToSpeech? = null
    private lateinit var tvTemperature: TextView
    private lateinit var tvDescription: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_weather)

        tvTemperature = findViewById(R.id.tvTemperature)
        tvDescription = findViewById(R.id.tvDescription)
        val btnBack = findViewById<Button>(R.id.btnBack)

        // Mock data
        val temp = "28°C"
        val desc = "Sunny with clear skies"
        tvTemperature.text = temp
        tvDescription.text = desc

        tts = TextToSpeech(this, this)

        btnBack.setOnClickListener {
            finish()
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.setLanguage(Locale.US)
            val weatherText = "The current weather is ${tvTemperature.text}. ${tvDescription.text}."
            tts?.speak(weatherText, TextToSpeech.QUEUE_FLUSH, null, null)
        } else {
            Log.e("WeatherActivity", "TTS Initialization failed")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.stop()
        tts?.shutdown()
    }
}
