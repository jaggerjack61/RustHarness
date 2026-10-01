//! Watches the keyboard while a turn runs so Esc (or Ctrl+C) can interrupt it.

use std::io::{self, IsTerminal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

use crate::cancel::CancelToken;

const POLL_INTERVAL: Duration = Duration::from_millis(50);

pub struct KeyListener {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    // Dropped last, after the thread has stopped reading.
    _mode: InputMode,
}

impl KeyListener {
    /// Start listening. Returns `None` when stdin is not an interactive terminal.
    ///
    /// The first Esc or Ctrl+C cancels `cancel`; a second Ctrl+C while the turn is
    /// still winding down quits the program.
    pub fn start(cancel: CancelToken) -> Option<Self> {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return None;
        }
        let mode = InputMode::enable()?;
        // Discard keys typed before this turn so a stray Esc cannot cancel it.
        while event::poll(Duration::ZERO).unwrap_or(false) {
            if event::read().is_err() {
                break;
            }
        }
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                match event::poll(POLL_INTERVAL) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(_) => break,
                }
                let Ok(Event::Key(key)) = event::read() else {
                    continue;
                };
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                let ctrl_c = key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char('c' | 'C'));
                if ctrl_c && cancel.is_cancelled() {
                    platform::restore();
                    // Show the cursor again; the turn hid it for the spinner.
                    println!("\x1b[?25h");
                    std::process::exit(130);
                }
                if ctrl_c || key.code == KeyCode::Esc {
                    cancel.cancel();
                }
            }
        });
        Some(Self {
            stop,
            thread: Some(thread),
            _mode: mode,
        })
    }
}

impl Drop for KeyListener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Terminal input mode that delivers keys immediately without echoing them,
/// restored on drop. Output processing is left untouched so normal printing works.
struct InputMode;

impl InputMode {
    fn enable() -> Option<Self> {
        platform::enable().then_some(Self)
    }
}

impl Drop for InputMode {
    fn drop(&mut self) {
        platform::restore();
    }
}

#[cfg(unix)]
mod platform {
    use std::sync::Mutex;

    static SAVED: Mutex<Option<libc::termios>> = Mutex::new(None);

    pub fn enable() -> bool {
        // SAFETY: termios is plain old data, and tcgetattr/tcsetattr only read or
        // write the struct we pass for the stdin file descriptor.
        unsafe {
            let mut termios: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut termios) != 0 {
                return false;
            }
            let original = termios;
            // ISIG off: Ctrl+C arrives as a key (interrupting the turn) instead of
            // killing the process and leaving the terminal in this mode.
            termios.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG | libc::IEXTEN);
            termios.c_cc[libc::VMIN] = 1;
            termios.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios) != 0 {
                return false;
            }
            *SAVED.lock().unwrap_or_else(|error| error.into_inner()) = Some(original);
        }
        true
    }

    pub fn restore() {
        let saved = SAVED
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(original) = saved {
            // SAFETY: restores the attributes previously read by tcgetattr.
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original);
            }
        }
    }
}

#[cfg(not(unix))]
mod platform {
    // On Windows, raw mode only changes the console input mode.
    pub fn enable() -> bool {
        crossterm::terminal::enable_raw_mode().is_ok()
    }

    pub fn restore() {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}
