# Is "co-visible in the same frame" the wrong notion for a static scene?

Every pose-correction method cited in this project (CoAlign, RoCo, HERO, NOPE,
V2X-PC) matches what two agents see in the same frame, because outdoors the
cars move. Here the objects do not: across 34,021 re-observations of the same
object id, displacement from first sighting is 0.0000 m. So an object the ego
saw earlier and the partner sees now is a valid correspondence.

Test split, 2,440 frames, 65 episodes (median 38 frames, robots step 0.08 m per
frame, so an episode covers roughly 3 m of travel).

| correspondences available | >=1 | >=2 | >=3 |
|---|---|---|---|
| same frame (what the field counts) | 74.6% | 63.8% | 53.1% |
| partner now vs ego memory | 80.7% | 68.3% | 57.9% |
| both robots remember | 85.1% | 76.1% | **67.4%** |

Of the frames with at most two same-frame correspondences, 30.5% reach three
with memory. Of the frames with none, 14.8% reach three and 41.3% reach one.
The gain grows through the episode: for frames 40+, 43.5% -> 76.7%.

Real but partial. The limit is the episode length: 3 m of travel does not add
many new objects. A deployment with longer runs would gain more; this dataset
cannot show how much.

Note the counts here use each agent's full visible list and so differ slightly
from `overlap_geometry` (46.9% of frames at <=2 here against 53.3% there, which
restricts to the ego's BEV window).
