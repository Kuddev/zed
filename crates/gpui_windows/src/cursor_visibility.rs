//! Pointer auto-hide state shared by the platform and every window.
//!
//! Win32 posts `WM_MOUSEMOVE` / `WM_NCMOUSEMOVE` without any physical motion
//! whenever it re-evaluates what lies under the pointer (a window is shown,
//! hidden or moved, the cursor image changes, ...). Restoring the pointer on
//! every such message brings back a pointer that typing has just hidden, so it
//! flickers on each keystroke. Following Microsoft's guidance for detecting
//! real motion, the pointer is shown again only once the reported screen
//! position differs from where it was hidden.

use std::sync::{Mutex, PoisonError};

/// Physical screen coordinates of the pointer.
pub(crate) type ScreenPoint = (i32, i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Visible,
    /// `at` is `None` when the hide position could not be queried; any
    /// subsequent move then restores the pointer, as before this state existed.
    Hidden {
        at: Option<ScreenPoint>,
    },
}

pub(crate) struct CursorVisibility {
    state: Mutex<State>,
}

impl CursorVisibility {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(State::Visible),
        }
    }

    pub(crate) fn is_visible(&self) -> bool {
        *self.lock() == State::Visible
    }

    /// Hides the pointer at `at`. Returns `false` when it was already hidden;
    /// the original hide position is kept so a stationary pointer stays hidden.
    pub(crate) fn hide(&self, at: Option<ScreenPoint>) -> bool {
        let mut state = self.lock();
        if *state != State::Visible {
            return false;
        }
        *state = State::Hidden { at };
        true
    }

    /// Reports a pointer move to `to`. Returns `true` when the pointer was
    /// hidden and has really moved, i.e. it must be shown again now.
    pub(crate) fn show_if_moved(&self, to: ScreenPoint) -> bool {
        let mut state = self.lock();
        match *state {
            State::Hidden { at: Some(at) } if at == to => false,
            State::Hidden { .. } => {
                *state = State::Visible;
                true
            }
            State::Visible => false,
        }
    }

    /// Marks the pointer visible without restoring it, e.g. when it left the
    /// window or the window was deactivated and another window owns it now.
    pub(crate) fn reset(&self) {
        *self.lock() = State::Visible;
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_move_at_hide_position_keeps_pointer_hidden() {
        let visibility = CursorVisibility::new();
        assert!(visibility.hide(Some((640, 360))));

        // Windows re-posts a mouse move at the unchanged position after each
        // keystroke; this must not bring the pointer back.
        assert!(!visibility.show_if_moved((640, 360)));
        assert!(!visibility.show_if_moved((640, 360)));
        assert!(!visibility.is_visible());

        assert!(visibility.show_if_moved((641, 360)));
        assert!(visibility.is_visible());
    }

    #[test]
    fn repeated_typing_keeps_the_first_hide_position() {
        let visibility = CursorVisibility::new();
        assert!(visibility.hide(Some((10, 20))));
        assert!(!visibility.hide(Some((30, 40))));

        assert!(!visibility.show_if_moved((10, 20)));
        assert!(visibility.show_if_moved((30, 40)));
    }

    #[test]
    fn unknown_hide_position_restores_on_any_move() {
        let visibility = CursorVisibility::new();
        assert!(visibility.hide(None));
        assert!(visibility.show_if_moved((0, 0)));
        assert!(visibility.is_visible());
    }

    #[test]
    fn moves_while_visible_and_reset_do_not_restore() {
        let visibility = CursorVisibility::new();
        assert!(!visibility.show_if_moved((5, 5)));

        assert!(visibility.hide(Some((5, 5))));
        visibility.reset();
        assert!(visibility.is_visible());
        assert!(!visibility.show_if_moved((6, 6)));

        // After a reset the pointer can be hidden again at a new position.
        assert!(visibility.hide(Some((6, 6))));
        assert!(!visibility.show_if_moved((6, 6)));
    }
}
