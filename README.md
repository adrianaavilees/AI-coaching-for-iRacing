README FOR THE PROJECT.

1. download data
2. add extra data
3. create final dataset
4. train
5. evaluation
6. UI streamlit run app/main.py

## Professional Streamlit interface

Run the presentation app with:

```powershell
streamlit run app/main.py
```

The interface is designed as a production-grade AI motorsport analytics platform for the final TFG defense. The main product flow is upload-first: the driver uploads a raw Garage61 CSV lap from the sidebar and the app runs the trained LSTM autoencoder analysis. The validated `models/eval_test_dani` session is available only as an optional presentation sample when no CSV is available; older evaluation folders are intentionally not exposed in the UI.

### Architecture

```text
app/
	main.py                    Streamlit page orchestration and session state
	models.py                  Typed view models for UI data
	settings.py                App constants separate from ML training config
	components/
		layout.py                KPI cards, header, AI coaching cards
		styles.py                Dark premium motorsport CSS theme
	services/
		data_service.py          Uploaded CSV inference adapter and optional sample loader
	visualization/
		plots.py                 Plotly track map, telemetry, heatmap, radar charts
	utils/
		reporting.py             Dependency-free PDF report export
```

### UX sections

- **Session Overview:** driver, track, car, lap count, best lap, delta vs expert, consistency score and main weaknesses.
- **Track Map:** interactive circuit trace with color-coded corner markers: green, yellow and red. Clicking a marker opens a detailed zone explanation in the inspector panel.
- **Telemetry Analysis:** synchronized Plotly overlays for speed, brake, throttle, steering and delta time with a highlighted selected corner.
- **AI Coaching:** Groq/template feedback rendered as coaching cards with issue, time loss, recommendation and severity.
- **Session Report:** strengths/weaknesses, corner ranking, radar chart and PDF download.
- **Advanced:** multi-lap consistency, sector analysis, driver-friendly focus timeline, scalability and engineering notes.

### Engineering best practices used

- Streamlit is only the presentation layer; pipeline and inference are isolated behind `app/services`.
- `st.cache_data` is used for JSON artifacts, NumPy arrays and uploaded CSV preprocessing.
- The app is prepared for progressive rendering: CSV parse first, deterministic anomaly detection second, Groq feedback in a background worker third.
- For the future full inference service, cache the PyTorch model/scaler/expert baseline with `st.cache_resource`, use `torch.inference_mode()`, prefer CUDA when available and store telemetry arrays as `float32`.
- Keep only small UI selections in `st.session_state`; avoid storing full duplicated DataFrames per widget.
- Use uploaded laps as the default workflow. Use `eval_test_dani` only as an optional demo sample for final presentation consistency.

### Future backend integration

The current upload path parses and aligns a raw Garage61 CSV, runs the trained autoencoder when the model artifacts are available, and falls back to a telemetry-only preview if inference cannot run. For the final production pipeline, keep extending `app/services/data_service.py` as the adapter that calls:

1. telemetry preprocessing and interpolation
2. scaler normalization
3. LSTM autoencoder reconstruction
4. reconstruction-error zone detection
5. `feedback_engine.generate_feedback(...)`
6. Groq feedback generation with template fallback

This keeps frontend/backend separation clean and allows the Streamlit interface to remain stable while the inference layer evolves into a background worker or FastAPI service.