"""Use the Raspberry Pi AI Camera to make the hexapod follow a cat."""

from follow_person import main


if __name__ == "__main__":
    raise SystemExit(main(default_target_label="cat"))
