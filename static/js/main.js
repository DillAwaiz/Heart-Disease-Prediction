/* ==================================================================
   Heart Disease Prediction - shared page behaviour
   Plain JavaScript, no frameworks.
     1. sidebar open/close on small screens
     2. flash messages fade away on their own
   ================================================================== */

document.addEventListener('DOMContentLoaded', function () {

  /* ---------------------------------------------------------------
     1. SIDEBAR TOGGLE (mobile)
     The sidebar is hidden off-screen under 900px wide. The burger
     button slides it in, and tapping the dark backdrop closes it.
     --------------------------------------------------------------- */
  var sidebar   = document.getElementById('sidebar');
  var toggleBtn = document.getElementById('sidebarToggle');
  var backdrop  = document.getElementById('sidebarBackdrop');

  function openSidebar() {
    if (!sidebar) return;
    sidebar.classList.add('open');
    if (backdrop) backdrop.classList.add('show');
  }

  function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove('open');
    if (backdrop) backdrop.classList.remove('show');
  }

  if (toggleBtn) {
    toggleBtn.addEventListener('click', function () {
      if (sidebar.classList.contains('open')) {
        closeSidebar();
      } else {
        openSidebar();
      }
    });
  }

  if (backdrop) {
    backdrop.addEventListener('click', closeSidebar);
  }

  // Escape key closes the menu
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeSidebar();
  });

  // Tapping a navigation link on mobile closes the menu straight away
  var navLinks = document.querySelectorAll('.sidebar .nav-link');
  for (var i = 0; i < navLinks.length; i++) {
    navLinks[i].addEventListener('click', function () {
      if (window.innerWidth <= 900) closeSidebar();
    });
  }

  // Returning to a wide screen should always show the sidebar normally
  window.addEventListener('resize', function () {
    if (window.innerWidth > 900) closeSidebar();
  });


  /* ---------------------------------------------------------------
     2. FLASH MESSAGES
     Close button works immediately; anything that is not an error
     disappears by itself after six seconds.
     --------------------------------------------------------------- */
  function dismiss(flash) {
    flash.classList.add('fade-out');
    setTimeout(function () {
      if (flash.parentNode) flash.parentNode.removeChild(flash);
    }, 400);
  }

  var flashes = document.querySelectorAll('.flash');
  for (var j = 0; j < flashes.length; j++) {
    (function (flash) {
      var closeBtn = flash.querySelector('.flash-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', function () { dismiss(flash); });
      }
      // Errors and warnings stay until the user dismisses them
      var isImportant = flash.classList.contains('flash-danger') ||
                        flash.classList.contains('flash-warning');
      if (!isImportant) {
        setTimeout(function () { dismiss(flash); }, 6000);
      }
    })(flashes[j]);
  }


  /* ---------------------------------------------------------------
     3. SHOW / HIDE PASSWORD
     Every password box gets a small eye button. Applied here rather
     than in each template so login, register and profile all get it.
     --------------------------------------------------------------- */
  var EYE_OPEN =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" ' +
    'stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>' +
    '<circle cx="12" cy="12" r="3"/></svg>';

  var EYE_SHUT =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" ' +
    'stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M17.9 17.9A10.1 10.1 0 0 1 12 20c-7 0-11-8-11-8a18.5 18.5 0 0 1 5.1-6"/>' +
    '<path d="M9.9 4.2A10.1 10.1 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.2 3.2"/>' +
    '<path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/><path d="M1 1l22 22"/></svg>';

  var passwordInputs = document.querySelectorAll('input[type="password"]');

  for (var p = 0; p < passwordInputs.length; p++) {
    (function (input) {
      // Wrap the input so the button can sit inside its right edge
      var wrap = document.createElement('div');
      wrap.className = 'password-wrap';
      input.parentNode.insertBefore(wrap, input);
      wrap.appendChild(input);

      var button = document.createElement('button');
      button.type = 'button';               // never submits the form
      button.className = 'password-toggle';
      button.innerHTML = EYE_OPEN;
      button.setAttribute('aria-label', 'Show password');
      button.setAttribute('title', 'Show password');
      wrap.appendChild(button);

      button.addEventListener('click', function () {
        var hidden = input.type === 'password';
        input.type = hidden ? 'text' : 'password';
        button.innerHTML = hidden ? EYE_SHUT : EYE_OPEN;
        var label = hidden ? 'Hide password' : 'Show password';
        button.setAttribute('aria-label', label);
        button.setAttribute('title', label);
        input.focus();
      });
    })(passwordInputs[p]);
  }


  /* ---------------------------------------------------------------
     4. CONFIRM BEFORE DESTRUCTIVE ACTIONS
     Any form carrying data-confirm asks first.
     --------------------------------------------------------------- */
  var confirmForms = document.querySelectorAll('form[data-confirm]');
  for (var k = 0; k < confirmForms.length; k++) {
    confirmForms[k].addEventListener('submit', function (e) {
      if (!window.confirm(this.getAttribute('data-confirm'))) {
        e.preventDefault();
      }
    });
  }


  /* ---------------------------------------------------------------
     5. SERVER-COMPUTED INLINE STYLES
     A handful of pages need a value the server calculated (a colour,
     a position, an image path) applied as inline CSS. Rather than
     writing Jinja expressions directly inside style="" - which code
     editors' embedded CSS checker misreads as broken CSS - the
     template puts the raw value in a data-* attribute and this runs
     once on load to turn it into real inline styles.
     --------------------------------------------------------------- */
  var bgUrlEls = document.querySelectorAll('[data-bg-url]');
  for (var b = 0; b < bgUrlEls.length; b++) {
    bgUrlEls[b].style.backgroundImage = "url('" + bgUrlEls[b].getAttribute('data-bg-url') + "')";
  }

  var leftPctEls = document.querySelectorAll('[data-left-pct]');
  for (var l = 0; l < leftPctEls.length; l++) {
    leftPctEls[l].style.left = leftPctEls[l].getAttribute('data-left-pct') + '%';
  }

  // Feature-importance bars: the template stores the width percentage in a
  // data attribute, and this applies it as a real width (no CSS in Jinja).
  var barEls = document.querySelectorAll('[data-bar-width]');
  for (var n = 0; n < barEls.length; n++) {
    barEls[n].style.width = barEls[n].getAttribute('data-bar-width') + '%';
  }

  var badgeColourEls = document.querySelectorAll('[data-badge-colour]');
  for (var m = 0; m < badgeColourEls.length; m++) {
    var colour = badgeColourEls[m].getAttribute('data-badge-colour');
    badgeColourEls[m].style.background = colour + '22';
    badgeColourEls[m].style.color = colour;
    badgeColourEls[m].style.border = '1px solid ' + colour;
  }

});
