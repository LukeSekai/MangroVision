# Workflow confirmation audit — October 22, 2026

Reviewed the current staff routes, planter entry/field application, public LIKE
booking form, and their reachable mutation controls. This source review and its
isolated checks did not submit live actions, send emails or restart the shared
backend. Publishing and deployment are separate from these validation results.

## Added or standardized

| Workflow | Confirmation |
| --- | --- |
| Assign Points | **Assign selected points?** shows the organization, activity and count. Cancel keeps the selection. Confirm submits the reviewed point IDs and activity. |
| Organization monitoring | **Save monitoring visit?** summarizes organization/date, deaths, located/unlocated counts, cause and health. Back keeps the observations and selection. |
| LGU site monitoring | **Save site monitoring?** identifies the individual site/date, selected deaths, cause and health. Back keeps the form. |
| Identify/correct death locations | **Save death locations?** reviews the location counts and effect on links; visit totals and dates are unchanged. Back keeps notes and selected IDs. |
| Annual planting goals | **Save planting goals?** identifies the year and both goals, including unset values. Cancel keeps the draft. |
| Activity completion/cancellation | **Mark activity completed?** / **Cancel activity?** explains read-only progress or cancellation notification. Back keeps the edited form. Ordinary schedule creation/editing keeps its existing form review. |
| Eligible unused appointment deletion | **Delete appointment?** replaces the native browser prompt, identifies the appointment/organization, and retains the existing deletion eligibility rules. |
| Activity code replacement/revocation | **Replace organization code?** / **Revoke organization code?** replaces the inline confirmation, names the activity, and explains that its current logins end while records remain. |
| Participant browser switching | **Switch access to this browser?** replaces the inline prompt and explains the effect on the other login and saved progress. |

The existing Modal component supplies loading/disabled states, focus containment,
Escape handling and consistent buttons. Monitoring reviews replace the current
form in the same dialog; they do not stack another modal on top of it. Form
state remains in the parent, so Back restores the draft. Ref-based submission
guards prevent repeated confirms from sending a second request while busy.

## Already covered and preserved

| Workflow | Existing safeguard |
| --- | --- |
| Staff sign-out and planter sign-out | Shared logout confirmation. |
| Staff credential changes/recovery | Explicit form validation and email-code verification. |
| Public appointment requests | Two form stages, contact/schedule fields, required consent and explicit Send request; idempotent retry and receipt. |
| Website approval/rescheduling/decline | Dedicated request-review dialog, requester-contact/agreement checks, decline reason and explicit final decision button. |
| Staff schedule approval | Approval dialog with activity, organization and agreed time. |
| High-tide schedule caution | Existing time-caution dialog; forecast and booking rules unchanged. |
| Partial/occupied/repeated image analysis | Existing contextual confirmation and server checks. |
| Save analysis/delete unused analysis | Existing save acknowledgment/review and guarded deletion confirmation. |
| Overlapping erosion save/edit | Existing **Save eroded zone?**, with a draft until confirmed; ordinary zone form saves remain direct. |
| Erosion/warning deletion and clearing erosion | Existing destructive confirmation dialogs; warning semantics unchanged. |
| Individual planting, skip and selected planting batch | Existing reviewed confirmation; existing success notifications and erosion-blocked modals. |
| Dead-location release/site area release/site deletion | Existing eligibility/version checks, preview and confirmation. |
| Point exports/report downloads | Existing scope/map/document previews and explicit download controls. |
| Filters, map tools, navigation, notification-read state, code reveal/copy | Direct controls; these do not need a destructive-action confirmation. |

Retired/unrouted screens such as PlanterActivityReport, the old Monitoring page,
SeedlingRecords and TideCalibration were not reactivated or used to introduce
obsolete workflows. LGU Sites, automatic project-site formation and shared
participant access retain their current behavior. No dependency, backend
authorization, database schema, model setting or calculation changed in this
confirmation pass.

## Verification

Interaction coverage uses isolated React/JSDOM fixtures and mocked responses.
It checks that opening a review and Cancel/Back do not write, forms and selected
locations survive returning, confirmation submits the reviewed scope/version,
failed saves retain the draft, and repeated confirmation is guarded. Existing
public booking, email decision forms, browser switching, used-analysis deletion,
zone actions and sign-out tests are retained. Desktop/narrow layout previews
use synthetic data only. This is not a live end-to-end field rehearsal.

Relevant checks from `MangroVision_New/client`:

```powershell
node --test src/components/Panel.test.js src/components/ActivityAndBooking.test.js src/components/AuthInteraction.test.js src/pages/LGUSites.test.js src/pages/SchedulingForm.test.js src/pages/Dashboard.test.js src/guides/tours.test.js src/guides/tourRuntime.test.js src/pages/SiteRelease.test.js src/pages/FieldApp.navigation.test.js
npm run build
```

See the [system user guide](system-user-guides.md) and
[thesis defense/demo workflow](thesis-defense-demo-workflow.md).
