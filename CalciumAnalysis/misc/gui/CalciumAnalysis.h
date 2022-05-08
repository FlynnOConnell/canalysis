#pragma once

#include "version.h"
#include "utils/parser.h"
#include "utils/io.h"
#include <filesystem>
#include "parser.h"


std::filesystem::path HomeDirectory;
std::filesystem::path DataDirectory;

#define HomeDirectory        %HOME%

{
public:
	// Help return lowercase if need be for epic games workshop integration 
	static std::string toLower(std::string str, bool changeInline = false);

private:
	// File Helpers
	static std::vector<std::filesystem::path> IterateDirectory(const std::filesystem::path& directory, const std::vector<std::string>& extensions, int depth = 0, int maxDepth = 3);
	static std::vector<std::filesystem::path> GetFilesFromDir(const std::filesystem::path& directory, int numExtension, ...);

	// Formatting workshop/freeplay map returns 
	static bool HasExtension(const std::string& fileExtension, const std::vector<std::string>& extensions);


	// Window settings
	bool isWindowOpen = false;
	bool isMinimized = false;
	std::string menuTitle = "CalciumAnalysis";
public:
	virtual void Render();
	virtual std::string GetMenuName();
	virtual std::string GetMenuTitle();
	virtual void SetImGuiContext(uintptr_t ctx);
	virtual bool ShouldBlockInput();
	virtual bool IsActiveOverlay();
	virtual void OnOpen();
	virtual void OnClose();

private:
	void renderInstantSettingsTab();
	void renderTrainingPacksTab();
	bool hooked = false;
	// Maps internal name to display name

};
