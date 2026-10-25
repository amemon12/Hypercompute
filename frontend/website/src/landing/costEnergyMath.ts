export type CostEnergyInputs = {
  outputTokens: number;
  localDecodeTps: number;
  cloudDecodeTps: number;
  localWatts: number;
  electricityPerKwh: number;
  gridKgPerKwh: number;
  cloudHourlyPrice: number;
};

export function calculateCostEnergy(input: CostEnergyInputs) {
  if (
    !Object.values(input).every(Number.isFinite) ||
    !Number.isInteger(input.outputTokens) || input.outputTokens < 2 ||
    input.localDecodeTps <= 0 || input.cloudDecodeTps <= 0 ||
    input.localWatts < 0 || input.electricityPerKwh < 0 ||
    input.gridKgPerKwh < 0 || input.cloudHourlyPrice < 0
  ) return null;

  // The runtime's recorded decode rate excludes the first generated token.
  const decodeTokens = input.outputTokens - 1;
  const localHours = decodeTokens / input.localDecodeTps / 3600;
  const cloudHours = decodeTokens / input.cloudDecodeTps / 3600;
  const localKwh = input.localWatts / 1000 * localHours;
  const localCost = localKwh * input.electricityPerKwh;
  const cloudCost = cloudHours * input.cloudHourlyPrice;
  const localElectricityPerHour = input.localWatts / 1000 * input.electricityPerKwh;

  return {
    decodeTokens,
    localHours,
    cloudHours,
    localKwh,
    localCost,
    cloudCost,
    costDifference: cloudCost - localCost,
    localKgCo2e: localKwh * input.gridKgPerKwh,
    breakEvenCloudTps: localElectricityPerHour > 0
      ? input.cloudHourlyPrice * input.localDecodeTps / localElectricityPerHour
      : null,
  };
}
