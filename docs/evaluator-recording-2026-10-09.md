# MangroVision evaluator introduction and recorded demonstration

**Audience:** Evaluators who will review or try the system.

**Target length:** About 9–10 minutes, including time to show each screen.

**Style:** Speak slowly, use simple English, and demonstrate one task at a time.

## 1. Prepare before recording

- Use a clearly labelled demonstration organization, activity, and planting area. Obtain an approved test dataset before saving planting or death records; those actions change the database.
- Prepare one drone photograph with usable metadata and one saved analysis of it. Use a saved result to avoid waiting through the full analysis during the video. Say when you switch to a previously processed result.
- Have a confirmed demonstration planting activity with a selected area and available points. Prepare a separate future activity at the same site to demonstrate that it remains editable.
- Prepare a demonstration planter account and a participant device. Sign in before recording or cut the credential and verification-code entry from the video.
- For the booking scene, use a request with an email address controlled by the team. Confirmation and update actions can send real messages. Hide login details and personal contact information in the recording.
- Confirm that the updated API, database migration, and frontend have been activated before demonstrating the new activity locks and booking emails. Source-code changes alone do not update the running testing server.
- Open the staff workspace, the planter page, and the LIKE landing page in separate tabs. The public planter address is <https://mangrovision-user-testing.vercel.app/field>; the LIKE website is the `/landing-page` route on the same frontend.
- Check microphone volume, browser zoom, map visibility, and the internet connection. Close private configuration files and unrelated tabs.

## 2. Recording flow

| Time | Scene | What to show |
|---|---|---|
| 0:00–0:55 | Introduction and problem | Title slide, then the staff workspace |
| 0:55–1:35 | Dashboard | Overall totals, species chart, and available reports |
| 1:35–3:10 | Image analysis | Upload settings, saved history, original image, detection overlay, GSD, and **View area in map** |
| 3:10–4:20 | Schedule and assignment | Confirmed activity, planting area, organization, activity selection, and point assignment |
| 4:20–5:20 | Planter workflow | Participant points, navigation, and one approved demonstration planting record |
| 5:20–6:05 | Protect recorded work | Return to the activity; show disabled planning fields, editable notes, and the separate future activity |
| 6:05–7:05 | Monitoring | A demonstration visit, death-location selection, brush start/stop, and saved visit history |
| 7:05–8:15 | LIKE booking and updates | Public request, staff review, confirmed schedule, and queued/sent email status |
| 8:15–9:15 | Technologies and evaluation | Brief explanation of the main technologies, followed by evaluation instructions |

Allow extra time for clicks and screen transitions. The timestamps are a guide, not a requirement to rush.

## 3. Ready-to-read speaking script

### Scene 1 — Introduction and problem

Good day. We are presenting **MangroVision**, a system designed to support mangrove planting and monitoring.

The problem we are addressing is how to organize planting locations and preserve reliable records of restoration work. When planting information, assignments, and monitoring results are kept separately, it becomes harder to see what was planted, where it was planted, and what needs attention.

MangroVision brings these records together. It uses drone images to help identify possible planting locations, applies spacing rules when generating planting points, and records planting and monitoring activities.

Our goals are to preserve records for better decisions, make planting work easier to coordinate, and support efforts to improve seedling survival through appropriate spacing and follow-up care. Spacing is one part of successful planting; site conditions and maintenance also matter.

### Scene 2 — Dashboard

This is the staff dashboard. It provides an overview of planting progress and the records available in the system.

Here, the seedling summary shows total seedlings, planted locations, dead locations, and the survival rate based on current recorded map statuses. The percentage compares locations marked planted with the total marked planted or dead. Monitoring visits are still needed to verify actual seedling condition.

The species chart and other reports help staff review the information without checking every point individually. We will use the demonstration records throughout this walkthrough.

### Scene 3 — Image analysis and GSD

In Image Analysis, staff select a drone image and review its settings and metadata. They also select the species used for the planting plan. The system detects canopy; the species selection is provided by the staff member.

For this recording, I will open a previously processed result from the analysis history. The image previews, analysis dates, and sorting options help users find the result they want to review.

The result shows the original image alongside the detection overlay. It includes canopy coverage, estimated plantable area, and the available planting points.

One important measurement here is **GSD**, or Ground Sampling Distance. GSD tells us how much ground distance one image pixel represents. For example, a GSD of one centimetre per pixel means that one pixel represents approximately one centimetre on the ground.

This scale helps convert image measurements into metres and square metres. It also helps apply spacing rules in ground units instead of simply counting pixels. The system uses camera and altitude information and checks image alignment against the georeferenced map.

I can select **View area in map** to locate this analysis on the planting map. The suggested locations remain subject to staff review and actual field conditions.

### Scene 4 — Schedule and assign planting points

Staff can create an activity and coordinate its date and time. The tide forecast provides guidance for planning, while staff confirm whether conditions and availability are suitable.

Once a tree-planting activity is confirmed, staff choose its planting area. In Planters, they select the organization, the project site, and the planting activity that will receive the points.

The activity selection connects the assignment to that specific activity. If more than one confirmed activity is available, staff must choose the correct one.

The selected points are distributed across participants using a zigzag arrangement. This helps divide the planting work while keeping the saved point locations and spacing.

### Scene 5 — Planter workflow

This is the planter side. A participant signs in to the organization's account and opens their assigned points.

The participant can choose the next available point and request navigation using the device's location. The map distinguishes road directions from the local guide to the planting point. The local guide indicates direction; participants should follow the actual marked planting lanes.

After planting, the participant records completion. I will do this only on an approved demonstration point. The record is saved so staff can review the updated planting status.

### Scene 6 — Protect recorded planting

Now I will return to the activity linked to that assignment.

Because planting has been recorded, the activity's planning details are locked. Staff cannot move the activity to another area, change its dates or expected counts, cancel it, or delete its planting history.

Contact details and notes can still be corrected, and progress can be updated. These corrections are recorded in the activity history.

The lock belongs to this activity's planting batch. A separate future activity at the same project site can still be planned and edited.

### Scene 7 — Monitoring

In Monitoring, staff record visits and observations for each organization. These records preserve the information needed to review planting progress over time.

When a visit reports dead seedlings, staff can identify their locations on the map. The brush tool starts selecting only after a click in the map. A second click stops the brush, allowing the user to move the pointer without selecting more points.

After saving the visit, the monitoring history remains available for review. This connects planting records with later observations and helps staff decide what follow-up work is needed.

### Scene 8 — LIKE booking and revised details

The LIKE landing website is an additional public feature. Visitors can submit an appointment request, which appears in the staff workspace for review.

A request is not automatically treated as an approved activity. Staff contact the requester, agree on the details, and confirm the appointment.

The system then records the schedule and queues the confirmation email. For tree planting, the confirmation can include planter access information.

If the agreed date, time, title, or expected participant count changes before planting starts, the email follows the revised details. A confirmation still waiting to send is refreshed. If the earlier confirmation was already sent, an update is queued. Cancelling an eligible activity queues a cancellation notice.

The existing planter account, password, and participant device registrations are preserved during these updates. Staff can see whether the message is queued, being sent, or sent.

### Scene 9 — Technologies and evaluation

MangroVision uses **React** for the interface and **FastAPI with Python** for the backend. **PostgreSQL with PostGIS** stores the records and supports geographic operations. Images and related files use private object storage.

The canopy analysis uses **Detectree2 with Detectron2**. Image matching uses **OpenCV**, including SIFT feature matching and RANSAC, to help align a drone image with the map. A hexagonal grid supplies evenly spaced candidate planting locations, which are checked against canopy buffers, restricted areas, erosion zones, and existing points.

These technologies were chosen to support an interactive map, canopy boundary detection, measurements in ground units, and records that remain connected across the workflow.

For your evaluation, please assess whether the screens and instructions are understandable, whether the workflow is easy to complete, whether errors explain how to correct a problem, and whether saved information can be found and reviewed.

This demonstration shows how MangroVision connects image analysis, planning, planting, monitoring, and preserved records. Thank you. We welcome your feedback and suggestions.

## 4. Short technical answers for evaluator questions

| Topic | Simple explanation | Why it is used here |
|---|---|---|
| React | Builds the interactive staff and planter screens. | Supports map interactions, forms, previews, and shared interface state. |
| FastAPI / Python | Handles requests and the system's backend logic. | Connects the interface, image-processing tools, validation, and database services. |
| PostgreSQL / PostGIS | Stores structured records and geographic data. | Supports transactions, relationships, boundaries, and spatial checks. |
| Private object storage | Stores image files and analysis assets separately from database records. | Keeps large files out of database rows and controls access through the application. |
| Detectree2 / Detectron2 | Uses an instance-segmentation model to predict canopy regions. | Canopy outlines support measuring coverage and excluding occupied areas. It does not automatically identify the planting species. |
| SIFT + RANSAC | Finds corresponding image features and rejects inconsistent matches when estimating alignment. | Helps place drone-image results on the georeferenced orthophoto. Alignment can still be uncertain. |
| GSD | Ground distance represented by one pixel. | Converts pixel measurements to approximate ground distances and areas. |
| Hexagonal grid | Generates regularly spaced candidate locations. | Supports consistent neighbour spacing; candidates are filtered before becoming available planting points. |
| Buffers and spatial checks | Remove candidates near canopy or inside excluded areas and check spacing against saved points. | Helps avoid occupied or restricted locations and duplicate planting suggestions. |
| Zigzag allocation | Selects geographic strips and divides them into balanced participant shares. | Organizes participant work without moving the saved planting coordinates. |
| Google Routes and local site guides | Shows available walking road directions and a separate guide inside the mapped site. | Helps participants reach the area and locate their assigned point. The local guide is not a surveyed walkway. |
| Tide forecast | Gives estimated water-level information for planning. | Helps staff assess timing alongside actual conditions; it is not proof of safe access or the cause of seedling death. |
| Brevo email delivery | Delivers queued booking messages through the configured backend transport. | Keeps the requester informed of confirmation, revision, or cancellation. “Sent” means accepted by the delivery service, not proof that the recipient read it. |

### GSD example

The camera-based calculation is:

**GSD = flight height above the ground × sensor width ÷ (focal length × image width in pixels)**

Use compatible units for sensor width and focal length. The result is metres per pixel when height is in metres. At **0.01 m/pixel**, a distance of 100 pixels represents approximately **1 metre**, and a pixel covers approximately **0.0001 m²** on level ground.

The effective analysis scale can be adjusted through image alignment. Incorrect altitude, camera metadata, image alignment, or uneven ground can affect measurements. GSD is an image scale, not a statement of GPS accuracy.

### Questions about survival

- **Goal:** Support better survival through spacing, suitable planning, preserved records, and follow-up care.
- **Current dashboard calculation:** Planted ÷ (Planted + Dead) × 100, using current recorded map statuses.
- **What the evaluation can establish:** Whether the system supports the workflow and presents records clearly. Demonstrating a biological survival improvement would require field observations over time and a suitable comparison.

## 5. Implementation notes for the team

The new activity protection applies to assignment batches linked to an activity. Migration `20261009_0015` adds this link and backfills only unique matches by organization, site, and assignment date. A read-only assessment on October 9 found 14 historical batches with no unique activity match, including 13 with planting history. Their records remain intact and are not automatically attached to an activity. Do not describe those historical batches as already covered by the new activity lock.

The migration and updated public testing API/frontend were activated on October 9. For scenes 6 and 8, use a newly linked demonstration batch and an approved demonstration recipient. No demonstration planting records or real test emails were created by the development regression checks.

Code references for the technical explanations:

- [GSD calculation](../canopy_detection/gsd_calculator.py)
- [Canopy detection](../canopy_detection/canopy_detector_hexagon.py) and [Detectree2 integration](../canopy_detection/detectree2_proper.py)
- [Image matching](../canopy_detection/ortho_matcher.py)
- [Zigzag allocation and navigation](zigzag-and-site-routing.md)
- [Activity protection](../mangrovision_db/activity_rules.py)
- [Booking email delivery](../mangrovision_db/appointment_email.py)
