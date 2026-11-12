Do not use Bulwark for:

- any commercial use of these weights (see [Licence and dataset terms](#licence-and-dataset-terms));
- decisions about individuals, such as employment, discipline or insurance pricing, or scoring and ranking drivers or workers;
- identifying or tracking people: the models have no identity outputs, and their embeddings must not be used to re-identify anyone;
- inferring emotions or intentions: the outputs describe physical states and visible actions;
- covert monitoring, or monitoring without the notice and legal basis the law requires;
- use as a certified safety component, a type-approved vehicle system or a medical device.

**What Bulwark does not do.** It does not estimate gaze, sleepiness on the Karolinska Sleepiness Scale (KSS), or impairment from alcohol or drugs. It is not an occupant monitoring system (OMS): it was trained to watch the driver, and it does not detect seat belts or passengers.

**Responsible use:**

- In research, check it on your own people, cameras and conditions before you rely on any output.
- Keep a human in the loop for any consequence for a person, and give people a way to contest it.
- Process video where it is captured when you can, and do not store face video by default.
- Tell the people being monitored, and get their consent where the law requires it.
- Keep to the out-of-scope list above.
