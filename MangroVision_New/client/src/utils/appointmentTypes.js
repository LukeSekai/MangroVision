export const APPOINTMENT_TYPES = [
  { value: 'field_visit', label: 'Field visit' },
  { value: 'clean_up_drive', label: 'Clean-up drive' },
  { value: 'tree_planting', label: 'Tree planting' },
];

export const appointmentTypeLabel = (type) => APPOINTMENT_TYPES.find((item) => item.value === type)?.label || 'Tree planting';
export const isPlantingAppointment = (activity) => !activity.appointment_type || activity.appointment_type === 'tree_planting';
