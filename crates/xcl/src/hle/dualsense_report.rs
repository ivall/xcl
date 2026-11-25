//! DualSense report layouts, without any I/O: the output report (rumble, adaptive triggers,
//! light bar, player and mute LEDs) over USB (0x02, 48 bytes). Byte offsets follow the layout
//! Linux's `hid-playstation` driver uses (`dualsense_output_report_common`); the trigger
//! effect blocks follow the publicly documented effect modes (0x05 off, 0x21 feedback,
//! 0x25 weapon, 0x26 vibration) with 10 zones along the trigger's travel.

/// One adaptive trigger's effect.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum TriggerEffect {
    /// no force: the trigger moves freely
    #[default]
    Off,
    /// resistance of `strength` (1..=8) from zone `start` (0..=9) to the end of the travel
    Feedback { start: u8, strength: u8 },
    /// a resistance from zone `start` (2..=7) that gives way at zone `end` (start+1..=8),
    /// like a gun's trigger
    Weapon { start: u8, end: u8, strength: u8 },
    /// the trigger vibrates from zone `start` (0..=9) on, `amplitude` 1..=8 at `frequency` Hz
    Vibration { start: u8, amplitude: u8, frequency: u8 },
}

impl TriggerEffect {
    /// The 11-byte effect block of the output report.
    pub fn block(self) -> [u8; 11] {
        let mut b = [0u8; 11];
        // the zones from `start` on, and each zone's 3-bit level (value - 1)
        let zones = |start: u8, level: u8| -> (u16, u32) {
            let (mut active, mut levels) = (0u16, 0u32);
            for z in start.min(9)..10 {
                active |= 1 << z;
                levels |= ((level.clamp(1, 8) - 1) as u32) << (3 * z as u32);
            }
            (active, levels)
        };
        match self {
            TriggerEffect::Off => b[0] = 0x05,
            TriggerEffect::Feedback { strength: 0, .. } | TriggerEffect::Vibration { amplitude: 0, .. } | TriggerEffect::Weapon { strength: 0, .. } => b[0] = 0x05,
            TriggerEffect::Feedback { start, strength } => {
                let (a, l) = zones(start, strength);
                b[0] = 0x21;
                b[1..3].copy_from_slice(&a.to_le_bytes());
                b[3..7].copy_from_slice(&l.to_le_bytes());
            }
            TriggerEffect::Weapon { start, end, strength } => {
                let start = start.clamp(2, 7);
                let end = end.clamp(start + 1, 8);
                let zones: u16 = (1 << start) | (1 << end);
                b[0] = 0x25;
                b[1..3].copy_from_slice(&zones.to_le_bytes());
                b[3] = strength.clamp(1, 8) - 1;
            }
            TriggerEffect::Vibration { start, amplitude, frequency } => {
                let (a, l) = zones(start, amplitude);
                b[0] = 0x26;
                b[1..3].copy_from_slice(&a.to_le_bytes());
                b[3..7].copy_from_slice(&l.to_le_bytes());
                b[9] = frequency;
            }
        }
        b
    }
}

/// Everything the host sets on the controller, sent as one output report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    /// compatible vibration (0..=255): the left, big motor and the right, small one. None:
    /// the actuators are left to the audio haptics (no rumble emulation).
    pub rumble: Option<(u8, u8)>,
    pub left_trigger: TriggerEffect,
    pub right_trigger: TriggerEffect,
    pub lightbar: [u8; 3],
    /// the five player LEDs, bit 0 the leftmost
    pub player_leds: u8,
    /// 0 high, 1 medium, 2 low
    pub led_brightness: u8,
    /// the microphone's LED
    pub mic_led: bool,
    /// fade out the blue light the controller shows until a host takes the light bar over
    /// (sent once, after connecting)
    pub lightbar_setup: bool,
    /// trigger effect strength attenuation, 0 (full) ..= 7
    pub trigger_attenuation: u8,
    /// rumble strength attenuation, 0 (full) ..= 7
    pub rumble_attenuation: u8,
}

impl Default for Output {
    fn default() -> Self {
        Output {
            rumble: Some((0, 0)),
            left_trigger: TriggerEffect::Off,
            right_trigger: TriggerEffect::Off,
            lightbar: [0, 0, 64],
            player_leds: 0x04,
            led_brightness: 0,
            mic_led: false,
            lightbar_setup: false,
            trigger_attenuation: 0,
            rumble_attenuation: 0,
        }
    }
}

/// size of the part common to USB and Bluetooth
pub const COMMON: usize = 47;

impl Output {
    /// The 47 bytes after the report id (USB) or after the sequence and tag (Bluetooth).
    pub fn common(&self) -> [u8; COMMON] {
        let mut c = [0u8; COMMON];
        // valid_flag0: right and left trigger effects; with rumble, compatible vibration and
        // haptics select (the motors come from the rumble bytes, not the audio channels)
        c[0] = 0x04 | 0x08;
        if let Some((left, right)) = self.rumble {
            c[0] |= 0x01 | 0x02;
            c[2] = right;
            c[3] = left;
        }
        // valid_flag1: mic mute LED, light bar, player LEDs, motor power level (attenuation)
        c[1] = 0x01 | 0x04 | 0x10 | 0x40;
        c[8] = self.mic_led as u8;
        c[10..21].copy_from_slice(&self.right_trigger.block());
        c[21..32].copy_from_slice(&self.left_trigger.block());
        c[36] = (self.rumble_attenuation & 7) | (self.trigger_attenuation & 7) << 4;
        if self.lightbar_setup {
            // valid_flag2: light bar setup control; setup 2: light out
            c[38] = 0x02;
            c[41] = 0x02;
        }
        c[42] = self.led_brightness.min(2);
        c[43] = self.player_leds & 0x1F;
        c[44..47].copy_from_slice(&self.lightbar);
        c
    }

    /// The USB output report 0x02.
    pub fn usb(&self) -> [u8; 48] {
        let mut r = [0u8; 48];
        r[0] = 0x02;
        r[1..].copy_from_slice(&self.common());
        r
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_effect_blocks() {
        assert_eq!(TriggerEffect::Off.block(), [0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        // feedback from zone 0 at the full strength: all 10 zones, level 7 in each
        let b = TriggerEffect::Feedback { start: 0, strength: 8 }.block();
        assert_eq!(b[0], 0x21);
        assert_eq!(u16::from_le_bytes([b[1], b[2]]), 0x3FF);
        assert_eq!(u32::from_le_bytes([b[3], b[4], b[5], b[6]]), 0x3FFF_FFFF);
        // from zone 5 at strength 2: zones 5..=9, level 1
        let b = TriggerEffect::Feedback { start: 5, strength: 2 }.block();
        assert_eq!(u16::from_le_bytes([b[1], b[2]]), 0x3E0);
        assert_eq!(u32::from_le_bytes([b[3], b[4], b[5], b[6]]), 0b001_001_001_001_001 << 15);
        // weapon: start and end zone bits, strength - 1
        assert_eq!(TriggerEffect::Weapon { start: 2, end: 5, strength: 8 }.block(), [0x25, 0x24, 0x00, 7, 0, 0, 0, 0, 0, 0, 0]);
        // out-of-range zones are brought into range
        assert_eq!(TriggerEffect::Weapon { start: 0, end: 0, strength: 1 }.block()[..4], [0x25, 0x0C, 0x00, 0]);
        // vibration: zones and amplitudes like feedback, the frequency at byte 9
        let b = TriggerEffect::Vibration { start: 0, amplitude: 4, frequency: 40 }.block();
        assert_eq!(b[0], 0x26);
        assert_eq!(u16::from_le_bytes([b[1], b[2]]), 0x3FF);
        assert_eq!(u32::from_le_bytes([b[3], b[4], b[5], b[6]]), 0b011_011_011_011_011_011_011_011_011_011);
        assert_eq!((b[7], b[8], b[9], b[10]), (0, 0, 40, 0));
        // zero strength is no effect
        assert_eq!(TriggerEffect::Vibration { start: 0, amplitude: 0, frequency: 40 }.block()[0], 0x05);
    }

    #[test]
    fn usb_output_report_layout() {
        let o = Output {
            rumble: Some((200, 100)),
            left_trigger: TriggerEffect::Feedback { start: 0, strength: 8 },
            right_trigger: TriggerEffect::Weapon { start: 2, end: 5, strength: 8 },
            lightbar: [0x11, 0x22, 0x33],
            player_leds: 0x04,
            led_brightness: 1,
            mic_led: true,
            lightbar_setup: true,
            trigger_attenuation: 2,
            rumble_attenuation: 1,
        };
        let r = o.usb();
        assert_eq!(r[0], 0x02);
        assert_eq!(r[1], 0x0F, "compatible vibration, haptics select, both triggers");
        assert_eq!(r[2], 0x55, "mic LED, light bar, player LEDs, motor power");
        assert_eq!((r[3], r[4]), (100, 200), "right (small) motor, then left (big)");
        assert_eq!(r[9], 1, "mute LED");
        assert_eq!(r[11..22], TriggerEffect::Weapon { start: 2, end: 5, strength: 8 }.block(), "right trigger");
        assert_eq!(r[22..33], TriggerEffect::Feedback { start: 0, strength: 8 }.block(), "left trigger");
        assert_eq!(r[37], 0x21, "attenuation: triggers high nibble, rumble low");
        assert_eq!((r[39], r[42]), (0x02, 0x02), "light bar setup: light out");
        assert_eq!((r[43], r[44]), (1, 0x04), "brightness, player LEDs");
        assert_eq!(r[45..48], [0x11, 0x22, 0x33], "light bar colour");
        // without rumble the actuators are left to the audio haptics
        let r = Output { rumble: None, ..o }.usb();
        assert_eq!(r[1], 0x0C);
        assert_eq!((r[3], r[4]), (0, 0));
        let r = Output { lightbar_setup: false, ..o }.usb();
        assert_eq!((r[39], r[42]), (0, 0));
    }
}
