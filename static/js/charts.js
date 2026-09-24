/* ==================================================================
   Heart Disease Prediction - charts
   One chart is enough for this project: the risk trend line shown on
   the dashboard. Chart.js is bundled in static/js/chart.min.js, so the
   application works with no internet connection.

   The chart reads its data from a <script type="application/json">
   block rendered by Jinja - no data is built by string concatenation.
   ================================================================== */

(function () {
  'use strict';

  /* Read a JSON payload embedded in the page. Returns null if missing. */
  function readData(elementId) {
    var el = document.getElementById(elementId);
    if (!el) return null;
    try {
      return JSON.parse(el.textContent);
    } catch (e) {
      return null;
    }
  }

  /* RISK TREND - a line of risk scores over time */
  function riskTrend(canvasId, dataId) {
    var data = readData(dataId);
    var canvas = document.getElementById(canvasId);
    if (!data || !data.length || !canvas || typeof Chart === 'undefined') return;

    new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: data.map(function (d) { return d.label; }),
        datasets: [{
          label: 'Risk probability (%)',
          data: data.map(function (d) { return d.value; }),
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239,68,68,.12)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#ef4444',
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { font: { size: 11 }, color: '#374151' } }
        },
        scales: {
          x: { ticks: { color: '#6b7280', font: { size: 9 }, maxRotation: 45 },
               grid: { display: false } },
          y: { min: 0, max: 100,
               ticks: { color: '#6b7280', font: { size: 10 },
                        callback: function (v) { return v + '%'; } },
               grid: { color: 'rgba(0,0,0,.05)' } }
        }
      }
    });
  }

  /* Expose the chart so templates can call it */
  window.HDPCharts = { riskTrend: riskTrend };
})();
